# Build targets for the pxq_llama container images.
#
#   VARIANT=cpu  docker buildx bake --load full swap
#   VARIANT=cu12 docker buildx bake --load full swap
#
# VARIANT selects the Containerfile under docker/ (anything starting with "cu"
# uses the CUDA one). Override CONTAINERFILE to point somewhere else.

variable "REPO_OWNER"   { default = "local" }
variable "VARIANT"      { default = "cpu" }
variable "BUILD_NUMBER" { default = "0" }
variable "CUDA_VERSION" { default = "12.6.2" }
variable "CUDA_DOCKER_ARCH" { default = "86;90" }
variable "USE_CCACHE"   { default = "true" }
variable "GGML_NATIVE"  { default = "ON" }

# Explicit Containerfile override. Empty means "derive it from VARIANT".
variable "CONTAINERFILE" { default = "" }

# Registry-side ccache. Only usable inside GitHub Actions, so it is opt-in:
# set GHA_CACHE=true in the workflow, leave it alone for local builds.
variable "GHA_CACHE" { default = "false" }

target "cache_settings" {
  cache-from = equal(GHA_CACHE, "true") ? ["type=gha,scope=ccache-${VARIANT}"] : []
  cache-to   = equal(GHA_CACHE, "true") ? ["type=gha,mode=max,scope=ccache-${VARIANT}"] : []
}

group "default" {
  targets = ["server", "full", "swap"]
}

target "settings" {
  context  = "."
  dockerfile = notequal(CONTAINERFILE, "") ? CONTAINERFILE : (
    substr(VARIANT, 0, 2) == "cu"
      ? "docker/pxq_llama-cuda.Containerfile"
      : "docker/pxq_llama-cpu.Containerfile"
  )
  inherits = ["cache_settings"]
  args = {
    BUILD_NUMBER     = "${BUILD_NUMBER}"
    CUDA_VERSION     = "${CUDA_VERSION}"
    CUDA_DOCKER_ARCH = "${CUDA_DOCKER_ARCH}"
    GGML_NATIVE      = "${GGML_NATIVE}"
    USE_CCACHE       = "${USE_CCACHE}"
  }
}

target "server" {
  inherits = ["settings"]
  target = "server"
  tags = [
    "ghcr.io/${REPO_OWNER}/pxq-llama-cpp:${VARIANT}-server-${BUILD_NUMBER}",
    "ghcr.io/${REPO_OWNER}/pxq-llama-cpp:${VARIANT}-server"
  ]
}

target "full" {
  inherits = ["settings"]
  target = "full"
  tags = [
    "ghcr.io/${REPO_OWNER}/pxq-llama-cpp:${VARIANT}-full-${BUILD_NUMBER}",
    "ghcr.io/${REPO_OWNER}/pxq-llama-cpp:${VARIANT}-full"
  ]
}

target "swap" {
  inherits = ["settings"]
  target = "swap"
  tags = [
    "ghcr.io/${REPO_OWNER}/pxq-llama-cpp:${VARIANT}-swap-${BUILD_NUMBER}",
    "ghcr.io/${REPO_OWNER}/pxq-llama-cpp:${VARIANT}-swap"
  ]
}
