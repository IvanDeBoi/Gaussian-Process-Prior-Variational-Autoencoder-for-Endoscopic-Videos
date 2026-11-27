import cv2
import os

def mp4_to_png_frames(
    video_path: str,
    output_dir: str = "frames",
    prefix: str = "frame_",
    start_index: int = 0,
    end_index: int = -1,
) -> int:
    """
    Extract all frames from an MP4 and save them as PNG images.

    Returns the number of frames written.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_nr_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_nr_of_frames_to_write = end_index - start_index + 1 if end_index != -1 else total_nr_of_frames
    print("Total nr of frames:", total_nr_of_frames)
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    frame_idx = start_index
    must_continue=True
    while must_continue:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        out_path = os.path.join(output_dir, f"{prefix}{frame_idx:06d}.png")
        # write as PNG
        if True or frame_idx % 100 == 0:
            print(f"Writing frame {frame_idx} ({frame_idx-start_index+1} / {total_nr_of_frames_to_write}) to {out_path}")
        cv2.imwrite(out_path, frame)
        frame_idx += 1
        if frame_idx > end_index and end_index != -1: # -1 means no end_index
            must_continue = False
    cap.release()
    return frame_idx - start_index

if __name__ == "__main__":
    video_path="../Data/kvasir-capsule-labeled-videos/labelled_videos/8a00709108cd4e2b.mp4"
    output_dir = "../Data/kvasir-capsule-labeled-videos/labelled_videos/8a00709108cd4e2b_frames/"
    start_index: int = 1000
    end_index: int = 2000
    n = mp4_to_png_frames(video_path, output_dir=output_dir, prefix="kvasir-capsule-labeled-videos_8a00709108cd4e2b_336_", start_index=start_index, end_index=end_index)
    print(f"Wrote {n} frames.")
