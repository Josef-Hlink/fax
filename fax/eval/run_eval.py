# -*- coding: utf-8 -*-

"""
Run closed loop evaluation of a model in the emulator.

Largely copied from https://github.com/ericyuegu/hal
"""

import random
import time
import traceback
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import List

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
    stop_event: EventType,
    enable_ffw: bool = True,
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
            enable_ffw=enable_ffw,
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
            for _ in iterator:
                if gamestate is None:
                    break

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
                controller_inputs = preprocessor.postprocess_preds(model_output)

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
    cfg: CFG,
    shared_batched_model_input_B: TensorDict,  # (n_workers,)
    shared_batched_model_output_B: TensorDict,  # (n_workers,)
    model_input_ready_flags: List[EventType],
    model_output_ready_flags: List[EventType],
    seq_len: int,
    stop_events: List[EventType],
    p1_weights_path: Path,
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
    preprocessor = Preprocessor(cfg)
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
    model = torch.compile(model, mode='default')
    with torch.no_grad():
        model(context_window_BL)
    logger.debug('Warmup step finished')

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

    i = 0
    while not all(event.is_set() for event in stop_events):
        iteration_start = time.perf_counter()

        wait_for_cpu_workers(timeout=cpu_flag_timeout)

        if all(event.is_set() for event in stop_events):
            break

        transfer_start = time.perf_counter()
        if i < seq_len:
            # while context window is not full, fill in from the left
            context_window_BL[:, i].copy_(shared_batched_model_input_B, non_blocking=True)
        else:
            # update the context window by rolling frame data left and adding new data on the right
            context_window_BL[:, :-1].copy_(context_window_BL[:, 1:].clone())
            context_window_BL[:, -1].copy_(shared_batched_model_input_B, non_blocking=True)
        transfer_time = time.perf_counter() - transfer_start

        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs_BL: TensorDict = model(context_window_BL)
        seq_idx = min(seq_len - 1, i)
        outputs_B: TensorDict = outputs_BL[:, seq_idx]
        inference_time = time.perf_counter() - inference_start

        writeback_start = time.perf_counter()
        # write last frame of model preds to shared buffer
        shared_batched_model_output_B.copy_(outputs_B)
        writeback_time = time.perf_counter() - writeback_start

        total_time = time.perf_counter() - iteration_start

        if i % 60 == 0:
            msg = f'Iteration {i}: Total: {total_time * 1000:.2f}ms '
            if debug:
                msg += f'(Update context: {transfer_time * 1000:.2f}ms, Inference: {inference_time * 1000:.2f}ms, Writeback: {writeback_time * 1000:.2f}ms)'
            _ = msg
            # logger.debug(msg)

        i += 1

        # signal to CPU workers that output is ready
        for output_flag in model_output_ready_flags:
            output_flag.set()

        # clear model_input_ready_flags for the next iteration
        for input_flag in model_input_ready_flags:
            input_flag.clear()


def run_closed_loop_evaluation(
    cfg: CFG,
    p1_weights_path: Path,
    player: Player = 'p1',
    enable_ffw: bool = False,
    debug: bool = False,
) -> None:
    mp.set_start_method('spawn', force=True)
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(cfg)
    n_workers = 2  # TODO: make configurable

    # create events to signal when cpu and gpu workers are ready
    model_input_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
    model_output_ready_flags: List[EventType] = [mp.Event() for _ in range(n_workers)]
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
            'seq_len': preprocessor.seq_len,
            'stop_events': stop_events,
            'p1_weights_path': p1_weights_path,
            'device': device,
            'debug': debug,
        },
    )
    gpu_process.start()

    matchups = [Matchup() for _ in range(n_workers)]
    base_replay_dir = cfg.paths.replays / 'cle' / f'{cfg.eval.p1_type}_vs_{cfg.eval.p2_type}'
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
                'stop_event': stop_events[i],
                'enable_ffw': enable_ffw,
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
    exposed_args = {'PATHS': 'weights replays', 'BASE': 'debug', 'EVAL': 'p1-type p2-type n-loops'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)

    # choose random candidate weights from specified run type
    p1_weights_path = random.choice(list((cfg.paths.weights / cfg.eval.p1_type).glob('*.pth')))

    # every loop will generate ~2x n_workers replays
    for i in range(cfg.eval.n_loops):
        logger.info(f'CLE loop {i + 1}/{cfg.eval.n_loops}')
        run_closed_loop_evaluation(
            cfg,
            p1_weights_path=p1_weights_path,
            player='p1',
            enable_ffw=False,
            debug=cfg.base.debug,
        )
