import sys, traceback
sys.path.insert(0, '.')
from inference.pipeline import InferencePipeline

def inspect_item(it):
    try:
        if isinstance(it, dict):
            return ('dict', list(it.keys()))
        else:
            keys = []
            for a in ('violation','frame_index','bbox','track_id','type'):
                if hasattr(it, a):
                    keys.append(a)
            return (type(it).__name__, keys or None)
    except Exception:
        return (type(it).__name__, 'inspect_failed')

def main():
    try:
        p = InferencePipeline(detector_device='cpu',
                              detector_name='fasterrcnn',
                              min_frames_for_violation=1,
                              rider_threshold=3)
        print('InferencePipeline created - consuming generator now...')
        gen = p.process_video('test_video.mp4')
        total_items = 0
        frames_seen = 0
        suspected_violations = 0
        first_item_shown = False

        for item in gen:
            total_items += 1
            if not first_item_shown:
                print('\n--- FIRST YIELDED ITEM REPR ---')
                try:
                    print(repr(item))
                except:
                    print('<unprintable>')
                print('--- INSPECT FIRST ITEM ---')
                print(inspect_item(item))
                first_item_shown = True

            try:
                if isinstance(item, dict) and ('frame' in item or 'frame_index' in item):
                    frames_seen += 1
                elif hasattr(item, 'frame_index') or hasattr(item, 'frame'):
                    frames_seen += 1
            except:
                pass

            s = str(item)
            if 'Violation' in s or 'violation' in s or 'violate' in s:
                suspected_violations += 1

            if total_items % 50 == 0:
                print(f'Processed {total_items} items | frames ~{frames_seen} | violations ~{suspected_violations}')

        print('\n=== FINISHED CONSUMING GENERATOR ===')
        print('Total yielded items:', total_items)
        print('Approx frames seen:', frames_seen)
        print('Suspected violations:', suspected_violations)

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
