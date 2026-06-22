#!/bin/bash
# Copy ARM64 binaries from parent binaries/ to Android jniLibs/
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$PROJ_DIR/../binaries"
JNI_DIR="$PROJ_DIR/app/src/main/jniLibs/arm64-v8a"
ASSETS_DIR="$PROJ_DIR/app/src/main/assets"

mkdir -p "$JNI_DIR" "$ASSETS_DIR"

echo "Copying .so libraries..."
cp "$BIN_DIR"/*.so "$JNI_DIR/"

echo "Copying llama-server binary..."
cp "$BIN_DIR"/llama-server "$ASSETS_DIR/"

echo "Done! Binaries ready for Android build."
echo "Files in jniLibs: $(ls "$JNI_DIR" | wc -l)"
