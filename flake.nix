{
  description = "fax — imitation learning agents for Super Smash Bros. Melee";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems =
        f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        # Python via uv; every dependency comes as a wheel into .venv, except
        # libmelee's pyenet, which compiles a C extension — hence mkShell
        # (with a C toolchain) rather than mkShellNoCC.
        default = pkgs.mkShell (
          {
            packages = [
              pkgs.python311
              pkgs.uv
            ];

            shellHook = ''
              # Use the nix-provided interpreter; don't let uv fetch its own.
              export UV_PYTHON="${pkgs.python311}/bin/python3.11"
              export UV_PYTHON_DOWNLOADS=never
            '';
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            # manylinux wheels (torch, numpy, pyarrow) link against libstdc++
            # and zlib, which NixOS doesn't keep in /usr/lib for them to find.
            # /run/opengl-driver/lib is where NixOS puts the NVIDIA driver's
            # libcuda, which torch loads at runtime.
            LD_LIBRARY_PATH =
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
              ]
              + ":/run/opengl-driver/lib";
          }
        );
      });
    };
}
