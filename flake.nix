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
                # /run/opengl-driver/lib provides libcuda.so on NixOS; other distros find it already.
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath linuxRuntimeLibs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              ''
              + ''
                uv sync --quiet
              '';
          };
        }
      );
    };
}
