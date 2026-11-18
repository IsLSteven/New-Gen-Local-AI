"""Tests for cross-platform llama.cpp asset selection across CUDA, Vulkan, macOS Metal, and CPU."""

from nglai.engine.installer import _select_release_assets

MOCK_RELEASES = [
    {"tag_name": "v1.0.0", "assets": []},
    {
        "tag_name": "b12498",
        "assets": [
            {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip", "browser_download_url": "https://x/cudart.zip"},
            {"name": "llama-b12498-bin-win-cuda-12.4-x64.zip", "browser_download_url": "https://x/win-cuda.zip"},
            {"name": "llama-b12498-bin-win-vulkan-x64.zip", "browser_download_url": "https://x/win-vk.zip"},
            {"name": "llama-b12498-bin-win-cpu-x64.zip", "browser_download_url": "https://x/win-cpu.zip"},
            {"name": "llama-b12498-bin-macos-arm64.tar.gz", "browser_download_url": "https://x/mac-arm.tar.gz"},
            {"name": "cudart-llama-b12498-bin-ubuntu-cuda-12.8-x64.tar.gz", "browser_download_url": "https://x/u-cudart.tgz"},
            {"name": "llama-b12498-bin-ubuntu-cuda-12.8-x64.tar.gz", "browser_download_url": "https://x/u-cuda.tgz"},
        ],
    },
]


def test_select_windows_cuda_skips_empty_release() -> None:
    tag, chosen = _select_release_assets(MOCK_RELEASES, "windows", "amd64", "cuda")
    assert tag == "b12498"
    assert len(chosen) == 2
    assert "cudart" in chosen[0]["name"]
    assert "win-cuda" in chosen[1]["name"]


def test_select_windows_vulkan_and_macos_arm64() -> None:
    _, vk = _select_release_assets(MOCK_RELEASES, "windows", "amd64", "vulkan")
    assert len(vk) == 1 and "vulkan" in vk[0]["name"]

    _, mac = _select_release_assets(MOCK_RELEASES, "darwin", "arm64", "metal")
    assert len(mac) == 1 and "macos-arm64" in mac[0]["name"]
