# Pin new versions for whatever is needed
# use the uv installer instead of copying the binaries from the distroless image, less headaches that way
FROM debian:trixie

WORKDIR /workspaces

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates
ADD https://astral.sh/uv/0.11.28/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
RUN uv python install 3.14.6 && uv python update-shell