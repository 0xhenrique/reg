{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    pkg-config
    rustc
    cargo
    rust-analyzer
    python3
    ruby
    gcc
    go
    nodejs
    lua
    bash
  ];
}
