# MacBoot — Custom Multi-Boot USB Tool for macOS

**Source:** Direct engineering session, Ventoy docs, Kali/Parrot docs
**Date:** 2026-04-09

## Problem
Ventoy (the standard multi-boot USB tool) has no macOS support. Developer confirmed it will never be added.

## Solution: MacBoot
Custom Ventoy alternative built entirely on macOS using:
- `diskutil` for GPT partitioning
- Docker (debian:bookworm) for building GRUB x86_64-efi binary via `grub-mkstandalone`
- Docker (debian:bookworm --privileged) for creating ext4 persistence filesystem
- `dd` for writing ISOs to dedicated partitions
- Real-Debrid API for converting torrent-only downloads to direct HTTP links

## Architecture
Each ISO gets its own dedicated GPT partition (no shared data partition). This bypasses the FAT32 4GB file size limit that would otherwise prevent large ISOs. GRUB EFI bootloader on a small FAT16 partition presents a boot menu and chainloads each ISO's native EFI bootloader.

## Persistence
Shared ext4 partition labeled "persistence" with `/home union` config. Both Kali Live and Parrot Security detect it automatically. Files saved in ~/  persist across reboots and are accessible from either distro.

## Partition Layout (32GB USB)
1. EFI (210 MB) — System
2. MACBOOT_EFI (199 MB) — GRUB bootloader + grub.cfg
3. KALI_LIVE (5.8 GB) — Kali Linux 2026.1 Live
4. KALI_PURPLE (4.9 GB) — Kali Purple 2026.1 Installer
5. PARROT (8.5 GB) — Parrot Security OS 7.1
6. PERSIST (12.1 GB) — Shared /home (ext4)

## Key Technical Findings
- GRUB does NOT support exFAT — eliminates exFAT as shared data partition
- macOS can't natively write NTFS or ext4 — eliminates those as macOS-writable shared partitions
- Dedicated partition per ISO sidesteps all filesystem compatibility issues
- Real-Debrid instantly caches popular Linux ISOs — torrent → HTTP in seconds
- `rdisk` (raw disk) on macOS gives ~22 MB/s write speed to USB
