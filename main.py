from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer


def main():
    # Read video
    video_frames = read_video("input_videos/video_1.mp4")

    # Initialize Tracker
    player_tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detector_model.pt")

    # Run Trackers
    player_tracks = player_tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/player_track_stubs.pkl"
    )

    ball_tracks = ball_tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/ball_track_stubs.pkl"
    )

    # Remove wrong ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    # Interpolate ball tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Draw Output
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()

    # Draw Object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
