import sys, traceback
sys.path.insert(0,'.')
from inference.pipeline import InferencePipeline

video = sys.argv[1] if len(sys.argv)>1 else 'test_video.mp4'
print('Using video:', video)
try:
    p = InferencePipeline(detector_device='cpu',
                          detector_name='fasterrcnn',
                          min_frames_for_violation=1,
                          rider_threshold=3)
    gen = p.process_video(video)
    count = 0
    for item in gen:
        count += 1
        if count <= 3:
            print('YIELD[',count,'] ->', repr(item)[:200])
    print('Done. Total yielded items (so far):', count)
except Exception:
    traceback.print_exc()
