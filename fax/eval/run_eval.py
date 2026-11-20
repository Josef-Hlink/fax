# -*- coding: utf-8 -*-

"""
Run closed loop evaluation of a model in the emulator.

Largely copied from https://github.com/ericyuegu/hal
"""

import random
import re
import time
import traceback
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import List

import attr
from tqdm import trange
import torch
import torch.multiprocessing as mp
from loguru import logger
from tensordict import TensorDict

from fax.config import CFG, create_parser, parse_args, setup_logger
from fax.model import Model
from fax.utils.constants import Player
from fax.utils.emulator_helper import EmulatorManager, find_open_udp_ports, Matchup
from fax.utils.schema import NP_TYPE_BY_COLUMN
from fax.utils.gamestate_utils import extract_eval_gamestate_as_tensordict
from fax.processing.preprocessor import Preprocessor


def cpu_worker(
    cfg: CFG,
    emulator_path: Path,
    iso_path: Path,
    shared_batched_model_input: TensorDict,
    shared_batched_model_output: TensorDict,
    rank: int,
    port: int,
    player: Player,
    matchup: Matchup,
    replay_dir: Path,
    preprocessor: Preprocessor,
    model_input_ready_flag: EventType,
    model_output_ready_flag: EventType,
    new_game_flag: EventType,
    stop_event: EventType,
    debug: bool = False,
) -> None:
    """
    CPU worker that preprocesses data, writes it into shared memory,
    and sends controller inputs to the emulator from model predictions.
    """

    setup_logger(Path(cfg.paths.logs) / Path(__file__).name.replace('.py', '.log'), debug=debug)
    with logger.contextualize(rank=rank):
        emulator_manager = EmulatorManager(
            udp_port=port,
            player=player,
            emulator_path=emulator_path,
            replay_dir=replay_dir,
            opponent_cpu_level=9,
            matchup=matchup,
            enable_ffw=cfg.eval.ffw,
            debug=debug,
        )
        try:
            gamestate_generator = emulator_manager.run_game(iso_path)
            gamestate = next(gamestate_generator)
            # skip first N frames to match starting frame offset from training sequence sampling
            logger.debug(
                f'Skipping {preprocessor.eval_warmup_frames} starting frames to match training distribution'
            )
            for _ in range(preprocessor.eval_warmup_frames):
                gamestate = next(gamestate_generator)

            # only show progress bar for one worker to avoid clutter
            iterator = trange(99999, desc=f'CPU worker 1') if rank == 1 else range(99999)
            prev_frame = gamestate.frame
            for _ in iterator:
                if gamestate is None:
                    break

                # the frame counter starts over with every game; tell the GPU worker, so the
                # agent does not start a game looking at the end of the previous one
                if gamestate.frame < prev_frame:
                    new_game_flag.set()
                prev_frame = gamestate.frame

                gamestate_td = extract_eval_gamestate_as_tensordict(gamestate)
                model_inputs = preprocessor.preprocess_inputs(gamestate_td, player)

                sharded_model_input: TensorDict = shared_batched_model_input[rank]
                # update our rank of the shared buffer with the last frame
                sharded_model_input.update_(model_inputs[-1], non_blocking=True)

                model_input_ready_flag.set()

                # wait for the output to be ready
                while not model_output_ready_flag.is_set() and not stop_event.is_set():
                    time.sleep(0.0001)  # sleep briefly to avoid busy waiting

                if stop_event.is_set():
                    break

                # read model output and postprocess
                model_output = shared_batched_model_output[rank].clone()
                controller_inputs = preprocessor.postprocess_preds(
                    model_output, temperature=cfg.eval.temperature
                )

                # send controller inputs to emulator, update gamestate
                gamestate = gamestate_generator.send((controller_inputs, None))

                # clear the output ready flag for the next iteration
                model_output_ready_flag.clear()
        except StopIteration:
            logger.debug(f'CPU worker {rank} episode complete.')
        except Exception as e:
            logger.error(
                f'CPU worker {rank} encountered an error: {e}\nTraceback:\n{"".join(traceback.format_tb(e.__traceback__))}'
            )
        finally:
            model_input_ready_flag.set()
            stop_event.set()
            logger.debug(f'CPU worker {rank} stopped')


def gpu_worker(
    stop_events: List[EventType], model_output_ready_flags: List[EventType], **kwargs
) -> None:
    """Runs the GPU worker, making sure the CPU workers are released when it exits or crashes."""
    try:
        _gpu_worker(
            stop_events=stop_events, model_output_ready_flags=model_output_ready_flags, **kwargs
        )
    except Exception as e:
        logger.error(
            f'GPU worker encountered an error: {e}\nTraceback:\n{"".join(traceback.format_tb(e.__traceback__))}'
        )
    finally:
        # without this, the CPU workers (and their emulators) wait for model outputs forever
        for stop_event in stop_events:
            stop_event.set()
        for output_flag in model_output_ready_flags:
            output_flag.set()


def _gpu_worker(
    cfg: CFG,
    shared_batched_model_input_B: TensorDict,  # (n_workers,)
    shared_batched_model_output_B: TensorDict,  # (n_workers,)
    model_input_ready_flags: List[EventType],
    model_output_ready_flags: List[EventType],
    new_game_flags: List[EventType],
    seq_len: int,
    stop_events: List[EventType],
    p1_weights_path: Path,
    stats_path: Path | None,
    device: torch.device | str,
    cpu_flag_timeout: float = 5.0,
    debug: bool = False,
) -> None:
    """
    GPU worker that batches data from shared memory, updates the context window,
    performs inference with model, and writes output back to shared memory.
    """
    setup_logger(Path(cfg.paths.logs) / Path(__file__).name.replace('.py', '.log'), debug=debug)
    torch.set_float32_matmul_precision('high')
    preprocessor = Preprocessor(cfg, stats_path)
    model = Model(preprocessor, cfg)
    with open(p1_weights_path, 'rb') as f:
        logger.debug(f'Loading model weights from {p1_weights_path}...')
        state_dict = torch.load(f, map_location='cpu')
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # stack along time dimension
    # shape: (n_workers, seq_len)
    inputs_BL: List[torch.Tensor] = [shared_batched_model_input_B for _ in range(seq_len)]  # type: ignore
    context_window_BL: TensorDict = torch.stack(inputs_BL, dim=-1).to(device)  # type: ignore
    logger.debug(f'Context window shape: {context_window_BL.shape}')

    # warmup CUDA graphs with dummy inputs
    logger.debug('Compiling model...')
    compiled_model = torch.compile(model, mode='default')
    try:
        with torch.no_grad():
            compiled_model(context_window_BL)
        model = compiled_model
        logger.debug('Warmup step finished')
    except Exception as e:
        # compilation needs a working C compiler and CUDA toolchain; inference is fast enough without
        logger.warning(f'Could not compile the model, running it uncompiled: {e}')

    def wait_for_cpu_workers(timeout: float = 5.0) -> None:
        # wait for all CPU workers to signal that data is ready
        flag_wait_start = time.perf_counter()
        for i, (input_flag, stop_event) in enumerate(zip(model_input_ready_flags, stop_events)):
            while not input_flag.is_set() and not stop_event.is_set():
                if not input_flag.is_set() and time.perf_counter() - flag_wait_start > timeout:
                    logger.warning(
                        f'CPU worker {i} input flag wait took too long, stopping episode'
                    )
                    input_flag.set()
                    stop_event.set()
                time.sleep(0.0001)  # sleep briefly to avoid busy waiting

    # longer timeout on init to allow for emulators to start
    wait_for_cpu_workers(timeout=30.0)

    # every worker has its own position in the context window, which goes back to the start
    # when its emulator begins a new game
    n_workers = len(stop_events)
    rows_B = torch.arange(n_workers, device=device)
    n_frames_B = torch.zeros(n_workers, dtype=torch.long, device=device)

    i = 0
    while not all(event.is_set() for event in stop_events):
        iteration_start = time.perf_counter()

        wait_for_cpu_workers(timeout=cpu_flag_timeout)

        if all(event.is_set() for event in stop_events):
            break

        for rank, new_game_flag in enumerate(new_game_flags):
            if new_game_flag.is_set():
                new_game_flag.clear()
                n_frames_B[rank] = 0
                logger.debug(f'Worker {rank} started a new game, context window cleared')

        transfer_start = time.perf_counter()
        # while a worker's context window is not full, fill it in from the left; after that,
        # roll its frame data left and add the new data on the right
        full_B = n_frames_B >= seq_len
        seq_idx_B = n_frames_B.clamp(max=seq_len - 1)
        for key, window in context_window_BL.items():
            if full_B.any():
                window[full_B, :-1] = window[full_B, 1:].clone()
            window[rows_B, seq_idx_B] = shared_batched_model_input_B[key].to(device)
        n_frames_B += 1
        transfer_time = time.perf_counter() - transfer_start

        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs_BL: TensorDict = model(context_window_BL)
        inference_time = time.perf_counter() - inference_start

        writeback_start = time.perf_counter()
        # write each worker's newest frame of model preds to shared buffer
        for key, output in outputs_BL.items():
            shared_batched_model_output_B[key].copy_(output[rows_B, seq_idx_B])
        writeback_time = time.perf_counter() - writeback_start

        total_time = time.perf_counter() - iteration_start

        if i % 60 == 0:
            msg = f'Iteration {i}: Total: {total_time * 1000:.2f}ms '
            if debug:
                msg += f'(Update context: {transfer_time * 1000:.2f}ms, Inference: {inference_time * 1000:.2f}ms, Writeback: {writeback_time * 1000:.2f}ms)'
            _ = msg
            # logger.debug(msg)

        i += 1

        # clear model_input_ready_flags for the next iteration; this has to happen before the
        # output flags are set: a CPU worker that is released first could already raise its
        # input flag again, only for us to wipe it and wait for a frame that never comes
        for input_flag, stop_event in zip(model_input_ready_flags, stop_events):
            if not stop_event.is_set():
                input_flag.clear()

        # signal to CPU workers that output is ready
        for output_flag in model_output_ready_flags:
            output_flag.set()


def run_closed_loop_evaluation(
    cfg: CFG,
    p1_weights_path: Path,
    stats_path: Path | None = None,
    player: Player = 'p1',
    debug: bool = False,
) -> None:
    mp.set_start_method('spawn', force=True)
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(cfg, stats_path)
    n_workers = cfg.eval.n_workers

    # create events to signal when cpu and gpu workers are ready
    model_input_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    model_output_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    new_game_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    # Create events to signal when emulator episodes end
    stop_events: List[EventType] = [mp.Event() for _ in range(n_workers)]

    # Share and pin buffers in CPU memory for transferring model inputs and outputs
    mock_framedata_L = mock_framedata_as_tensordict(preprocessor.trajectory_sampling_len)
    # Store only a single time step to minimize copying
    mock_model_inputs_ = preprocessor.preprocess_inputs(mock_framedata_L, player)[-1]
    # batch_size == n_workers
    shared_batched_model_input_B: TensorDict = torch.stack(
        [mock_model_inputs_ for _ in range(n_workers)],
        dim=0,  # type: ignore
    )
    shared_batched_model_input_B = share_and_pin_memory(shared_batched_model_input_B)
    preds: list[torch.Tensor] = [preprocessor.mock_preds_as_tensordict() for _ in range(n_workers)]  # type: ignore
    shared_batched_model_output_B: TensorDict = torch.stack(preds, dim=0)  # type: ignore
    shared_batched_model_output_B = share_and_pin_memory(shared_batched_model_output_B)

    gpu_process: mp.Process = mp.Process(
        target=gpu_worker,
        kwargs={
            'cfg': cfg,
            'shared_batched_model_input_B': shared_batched_model_input_B,
            'shared_batched_model_output_B': shared_batched_model_output_B,
            'model_input_ready_flags': model_input_ready_flags,
            'model_output_ready_flags': model_output_ready_flags,
            'new_game_flags': new_game_flags,
            'seq_len': preprocessor.seq_len,
            'stop_events': stop_events,
            'p1_weights_path': p1_weights_path,
            'stats_path': stats_path,
            'device': device,
            'debug': debug,
        },
    )
    gpu_process.start()

    matchups = [
        Matchup('BATTLEFIELD', cfg.eval.character, cfg.eval.character) for _ in range(n_workers)
    ]
    # for example, ~/Data/fax/replays/eval/MARTH/FvX-ft_vs_cpu
    base_replay_dir = (
        cfg.paths.replays
        / 'eval'
        / cfg.eval.character
        / f'{cfg.eval.p1_type}_vs_{cfg.eval.p2_type}'
    )
    base_replay_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f'Replays will be saved to {base_replay_dir}')

    cpu_processes: List[mp.Process] = []
    ports = find_open_udp_ports(n_workers)
    for i, matchup in enumerate(matchups):
        replay_dir = base_replay_dir / f'{i:03d}'
        replay_dir.mkdir(exist_ok=True, parents=True)
        p: mp.Process = mp.Process(
            target=cpu_worker,
            kwargs={
                'cfg': cfg,
                'emulator_path': cfg.paths.exe,
                'iso_path': cfg.paths.iso,
                'shared_batched_model_input': shared_batched_model_input_B,
                'shared_batched_model_output': shared_batched_model_output_B,
                'rank': i,
                'port': ports[i],
                'player': player,
                'matchup': matchup,
                'replay_dir': replay_dir,
                'preprocessor': preprocessor,
                'model_input_ready_flag': model_input_ready_flags[i],
                'model_output_ready_flag': model_output_ready_flags[i],
                'new_game_flag': new_game_flags[i],
                'stop_event': stop_events[i],
                'debug': debug,
            },
        )
        cpu_processes.append(p)
        p.start()

    gpu_process.join()

    for p in cpu_processes:
        p.join()

    # clean up replay dir
    flatten_replay_dir(base_replay_dir)
    logger.info('Closed loop evaluation complete')


def matchup_from_run_type(run_type: str) -> str:
    """Extract the training matchup from a run type, e.g. '1k-XvF-ft' -> 'XvF'."""
    match = re.fullmatch(r'[^-]+-([FX]v[FX])(-ft)?', run_type)
    assert match is not None, f"Run type '{run_type}' is not like <n-epochs>-<matchup>[-ft]"
    return match.group(1)


def mock_framedata_as_tensordict(seq_len: int) -> TensorDict:
    """Mock `seq_len` frames of gamestate data."""
    return TensorDict({k: torch.ones(seq_len) for k in NP_TYPE_BY_COLUMN}, batch_size=(seq_len,))


def share_and_pin_memory(tensordict: TensorDict) -> TensorDict:
    """
    Move tensordict to both shared and pinned memory.

    https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
    """
    tensordict.share_memory_()

    cudart = torch.cuda.cudart()
    if cudart is None:
        return tensordict

    for tensor in tensordict.flatten_keys().values():
        assert isinstance(tensor, torch.Tensor)
        cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        assert tensor.is_shared()
        assert tensor.is_pinned()

    return tensordict


def flatten_replay_dir(replay_dir: Path) -> None:
    """Copy all files to base replay dir and clean up subdirs."""
    for file in replay_dir.glob('**/*.slp'):
        target_path = replay_dir / file.name
        counter = 1
        while target_path.exists():
            stem = file.stem
            target_path = replay_dir / f'{stem}_{counter}{file.suffix}'
            counter += 1

        try:
            file.replace(target_path)
        except OSError as e:
            logger.warning(f'Failed to move replay file {file}: {e}')

    for directory in sorted(replay_dir.glob('**/'), key=lambda x: len(str(x)), reverse=True):
        if directory != replay_dir:
            try:
                directory.rmdir()
            except OSError as e:
                logger.warning(f'Failed to remove directory {directory}: {e}')


if __name__ == '__main__':
    exposed_args = {
        'PATHS': 'iso exe mds weights replays',
        'BASE': 'debug',
        'EVAL': '*',
    }
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)

    # use the requested run's weights, or else a random candidate from the specified run type
    candidates = sorted((cfg.paths.weights / cfg.eval.p1_type).glob('*.pth'))
    assert candidates, f'No weights found in {cfg.paths.weights / cfg.eval.p1_type}'
    if cfg.eval.p1_run:
        p1_weights_path = cfg.paths.weights / cfg.eval.p1_type / f'{cfg.eval.p1_run}.pth'
        assert p1_weights_path in candidates, f'{p1_weights_path} does not exist'
    else:
        p1_weights_path = random.choice(candidates)
    logger.info(f'Evaluating {p1_weights_path.name} (1 of {len(candidates)} candidates)')

    # inputs have to be normalized with the stats the agent was trained with, which training
    # saves next to the weights; for older weights, fall back on the bucket of the matchup in
    # the run type (not on whatever the default matchup is)
    stats_path: Path | None = p1_weights_path.with_suffix('.stats.json')
    if stats_path.exists():
        logger.info(f'Normalizing inputs with {stats_path.name}')
    else:
        stats_path = None
        matchup = matchup_from_run_type(cfg.eval.p1_type)
        cfg = attr.evolve(cfg, training=attr.evolve(cfg.training, matchup=matchup))
        logger.warning(f'No stats saved with these weights, using the {matchup} stats in the MDS')

    if cfg.eval.ffw:
        # measured by sending known inputs: the game does not wait for them in this mode, so
        # roughly 1 in 3 is skipped and another 1 in 3 is applied for two frames
        logger.warning('Fast-forward is not frame-accurate: the agent plays with dropped inputs')

    # every loop will generate ~2x n_workers replays
    for i in range(cfg.eval.n_loops):
        logger.info(f'CLE loop {i + 1}/{cfg.eval.n_loops}')
        run_closed_loop_evaluation(
            cfg,
            p1_weights_path=p1_weights_path,
            stats_path=stats_path,
            player='p1',
            debug=cfg.base.debug,
        )
