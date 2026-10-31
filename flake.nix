{
  description = "starship: Warp-based rocket landing simulator";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          python = pkgs.python313;
          # PyPI wheels (warp-lang, pygame-ce, rerun-sdk) dlopen these at runtime on Linux.
          linuxRuntimeLibs = with pkgs; [
            stdenv.cc.cc.lib
            zlib
            libGL
            vulkan-loader
            wayland
            libxkbcommon
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.just
              pkgs.tmux
            ];

            env = {
              UV_PYTHON = "${python}/bin/python3.13";
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook =
              pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
                # /run/opengl-driver/lib provides libcuda on NixOS. On Ubuntu, expose only the
                # driver libraries: adding its whole system lib directory would mix glibc builds.
                cuda_driver_path=/run/opengl-driver/lib
                if [ ! -e "$cuda_driver_path/libcuda.so.1" ] && [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
                  cuda_driver_path="$PWD/.direnv/cuda-driver"
                  mkdir -p "$cuda_driver_path"
                  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$cuda_driver_path/libcuda.so.1"
                  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so "$cuda_driver_path/libcuda.so"
                fi
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath linuxRuntimeLibs}:$cuda_driver_path''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              ''
              + ''
                uv sync --quiet
              '';
          };
        }
      );
    };
}
