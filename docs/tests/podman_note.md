I'm using library testcontainers what is generally meant to work with Docker, but I'm using Podman.

It's work fine with Podman, but it is some caveats.
If you are using Docker, probably instruction from Podman Desktop work fine:
https://podman-desktop.io/tutorial/testcontainers-with-podman

But if you don't, you have to also make an imitation for docker socket, by:
sudo ln -sf "$XDG_RUNTIME_DIR/podman/podman.sock" /var/run/docker.sock

Be aware, do it only if you do not have Docker installed, that probably can break your Docker installation.

One test case explicitly use podman, via subprocess

I tried to avoid that as much, as it is possible, but that test need hugging face token to register, and I do not want to
add token to image, so I used build secrets:
https://docs.docker.com/build/building/secrets/

Unfortunately it's look like testcontainers using Python library docker, what do not support that kind of secret.
Only runtime secrets are supported:
Docker note:
https://docs.docker.com/reference/cli/docker/secret/create/
Docker Python library secret docks:
https://docker-py.readthedocs.io/en/7.0.0/secrets.html
There is also no option to add secrets in building command:
https://docker-py.readthedocs.io/en/stable/images.html

Due to its lack of feature I was coerced to directly use podman via subprocess.