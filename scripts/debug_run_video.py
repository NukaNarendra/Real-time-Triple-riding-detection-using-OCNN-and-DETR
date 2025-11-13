import sys, traceback
sys.path.insert(0, '.')
from inference.pipeline import InferencePipeline

def main():
    try:
        # relaxed settings for testing
        p = InferencePipeline(detector_device='cpu',
                              detector_name='fasterrcnn',
                              min_frames_for_violation=1,
                              rider_threshold=3)
        print('??  InferencePipeline created:', p)
        print('??  Starting headless processing of test_video.mp4 (verbose)...')
        res = p.process_video('test_video.mp4', save_output=True, verbose=True)
        print('\n? PROCESS SUMMARY:', res)
    except Exception as e:
        print('\n? Exception while running pipeline:')
        traceback.print_exc()

if __name__ == "__main__":
    main()
