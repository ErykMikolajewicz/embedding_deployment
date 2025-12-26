I'm using library testcontainers what is generally meant to work with Docker, but I'm using Podman.

It's work fine with Podman, but it is some caveats.
If you are using Docker, probably instruction from Podman Desktop work fine:
https://podman-desktop.io/tutorial/testcontainers-with-podman

But if you don't, you have to also make an imitation for docker socket, by:
sudo ln -sf "$XDG_RUNTIME_DIR/podman/podman.sock" /var/run/docker.sock

Be aware, do it only if you do not have Docker installed, that probably can break your Docker installation.
