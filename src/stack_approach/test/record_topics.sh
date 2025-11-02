#!/usr/bin/env bash
set -e

# --- Usage check ---
if [ $# -ne 1 ]; then
  echo "Usage: $0 <bag_name>"
  echo "Example: $0 bag_test"
  exit 1
fi

# --- Variables ---
BAG_BASE="$1"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
BAG_DIR="${BAG_BASE}_${TIMESTAMP}.bag"

# --- Topics to record ---
TOPICS=(
  "/camera/color/image_raw/compressed"
  "/camera/aligned_depth_to_color/image_raw"
  "/camera/color/camera_info"
)

echo "-------------------------------------"
echo "📦 Recording ROS 2 bag: ${BAG_DIR}"
echo "🕒 Timestamp: ${TIMESTAMP}"
echo "📝 Topics:"
for t in "${TOPICS[@]}"; do
  echo "   - $t"
done
echo "-------------------------------------"

# --- Record command ---
ros2 bag record "${TOPICS[@]}" -o "${BAG_DIR}"
