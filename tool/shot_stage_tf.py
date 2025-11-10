# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import matplotlib.pyplot as plt

# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean

# # ====================== CONFIG / CONSTANTS ======================

# NEUTRAL_COLOR = (0, 255, 0)
# PRE_SHOT_COLOR = (0, 165, 255)
# FOLLOW_THROUGH_COLOR = (255, 255, 0)

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# RIGHT_LANDMARKS = [12, 14, 16]
# LEFT_LANDMARKS  = [11, 13, 15]

# pose = mp_pose.Pose(
#     model_complexity=2,
#     static_image_mode=False,
#     smooth_landmarks=True,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# # ====================== POSE & ANGLE UTILS ======================

# def get_3d_point(landmarks, index, width, height):
#     if index >= len(landmarks) or landmarks[index].visibility < 0.5:
#         return None
#     return np.array([
#         landmarks[index].x * width,
#         landmarks[index].y * height,
#         landmarks[index].z
#     ])

# def calculate_3d_angle(a, b, c):
#     if a is None or b is None or c is None:
#         return None
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
#     if denom < 1e-5:
#         return None
#     cosine_angle = np.dot(ba, bc) / denom
#     return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# def get_arm_state(landmarks, width, height):
#     right_shoulder = get_3d_point(landmarks, 12, width, height)
#     right_elbow    = get_3d_point(landmarks, 14, width, height)
#     right_wrist    = get_3d_point(landmarks, 16, width, height)
#     left_wrist     = get_3d_point(landmarks, 15, width, height)
#     left_hip       = get_3d_point(landmarks, 23, width, height)
#     right_hip      = get_3d_point(landmarks, 24, width, height)

#     if (right_wrist is not None and left_wrist is not None and
#         left_hip is not None and right_hip is not None and
#         right_shoulder is not None):
#         waist_y = (left_hip[1] + right_hip[1]) / 2.0
#         avg_wrist_y = (right_wrist[1] + left_wrist[1]) / 2.0
#         dist_wrists = np.linalg.norm(right_wrist - left_wrist)
#         if (dist_wrists < 0.15 * width and avg_wrist_y < waist_y
#             and right_wrist[1] > right_shoulder[1]):
#             return "pre_shot"

#     if right_wrist is not None and right_shoulder is not None:
#         if right_shoulder[1] > right_wrist[1]:
#             return "follow_through"

#     if right_shoulder is not None and right_wrist is not None:
#         if right_wrist[1] > right_shoulder[1]:
#             return "neutral"

#     return "neutral"

# # ====================== CAPTURE & STORE SHOT ANGLES ======================

# def record_shot(prompt="Recording Shot"):
#     cap = cv2.VideoCapture(0)
#     print(f"\n--- {prompt}: Please perform your shot. Press 'q' to quit early. ---")
#     shot_angles = []
#     previous_stage = "neutral"
#     start_time = None
#     last_print_time = time.time()
#     done = False

#     while not done:
#         ret, frame = cap.read()
#         if not ret:
#             print("No frame captured. Exiting.")
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)
        
#         h, w, _ = frame.shape

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             state = get_arm_state(landmarks, w, h)

#             if state == "pre_shot":
#                 current_color = PRE_SHOT_COLOR
#                 status_text   = "Pre-Shot"
#             elif state == "follow_through":
#                 current_color = FOLLOW_THROUGH_COLOR
#                 status_text   = "Follow Through"
#             else:
#                 current_color = NEUTRAL_COLOR
#                 status_text   = "Neutral"
            
#             cv2.putText(frame, status_text, (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#             mp_drawing.draw_landmarks(
#                 frame,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(
#                     color=current_color, thickness=2, circle_radius=3
#                 ),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(
#                     color=current_color, thickness=2
#                 )
#             )

#             # Compute angles
#             right_shoulder = get_3d_point(landmarks, 12, w, h)
#             right_elbow    = get_3d_point(landmarks, 14, w, h)
#             right_wrist    = get_3d_point(landmarks, 16, w, h)

#             elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
#             wrist_angle = calculate_3d_angle(right_elbow, right_wrist,
#                                              get_3d_point(landmarks, 20, w, h))
#             arm_angle   = calculate_3d_angle(get_3d_point(landmarks, 11, w, h),
#                                              right_shoulder, right_elbow)

#             if state in ["pre_shot", "follow_through"]:
#                 current_time = time.time()
#                 if start_time is None:
#                     start_time = current_time
#                 if current_time - last_print_time >= 0.1:
#                     elapsed_time = (current_time - start_time)
#                     shot_angles.append((state, elapsed_time,
#                                         elbow_angle, wrist_angle, arm_angle))
#                     last_print_time = current_time

#             if state != previous_stage:
#                 if state == "pre_shot":
#                     start_time = time.time()
#                     shot_angles = []
#                     last_print_time = start_time
#                 elif state == "neutral":
#                     if shot_angles:
#                         phases = [x[0] for x in shot_angles]
#                         if "pre_shot" in phases and "follow_through" in phases:
#                             print("\nShot Completed.")
#                             done = True
#                 previous_stage = state

#         cv2.imshow(prompt, frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Quit early.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return shot_angles

# # ====================== OVERALL FORM MEASURE ======================

# def compute_overall_form(e, w, a):
#     angles = []
#     if e is not None:
#         angles.append(e)
#     if w is not None:
#         angles.append(w)
#     if a is not None:
#         angles.append(a)
#     if len(angles) == 0:
#         return None
#     return sum(angles) / len(angles)  # e.g. simple average

# def extract_form_series(shot_data):
#     times = []
#     form_vals = []
#     for entry in shot_data:
#         t  = entry[1]
#         e  = entry[2]
#         w  = entry[3]
#         a  = entry[4]
#         measure = compute_overall_form(e, w, a)
#         if measure is not None:
#             times.append(t)
#             form_vals.append(measure)
#     times = np.array(times, dtype=np.float32)
#     form_vals = np.array(form_vals, dtype=np.float32)
#     return times, form_vals

# # ====================== SHIFT USER TIME SO START EVENTS ALIGN ======================

# def align_start_event(bench_times, user_times, bench_events, user_events, start_event_name="Start"):
#     """
#     If both shots have an event named 'Start', shift user_times so that
#     user_start_time lines up with bench_start_time.
#     :param bench_events: list of (evt_name, evt_time)
#     :param user_events:  list of (evt_name, evt_time)
#     :returns user_times_offset
#     """
#     bench_start_time = None
#     user_start_time  = None

#     # find the time for 'Start' in bench_events
#     for (name, t) in bench_events:
#         if name.lower() == start_event_name.lower():
#             bench_start_time = t
#             break

#     for (name, t) in user_events:
#         if name.lower() == start_event_name.lower():
#             user_start_time = t
#             break

#     if bench_start_time is not None and user_start_time is not None:
#         offset = bench_start_time - user_start_time
#         user_times_offset = user_times + offset
#         return user_times_offset
#     else:
#         return user_times  # no shift if we didn't find matching events

# # ====================== CLOSENESS MEASURE (CLAMPED AT 100) ======================

# def compute_user_closeness(bench_form, user_form, path):
#     alpha = 2.0
#     user_map = {}
#     for (i, j) in path:
#         if j not in user_map:
#             user_map[j] = []
#         user_map[j].append(i)

#     user_closeness = np.zeros_like(user_form)
#     for j in range(len(user_form)):
#         if j in user_map:
#             i_list = user_map[j]
#             i_mid = i_list[len(i_list)//2]
#             diff = abs(user_form[j] - bench_form[i_mid])
#             closeness_j = 100 - alpha * diff
#             closeness_j = max(0, min(100, closeness_j))
#         else:
#             closeness_j = 100
#         user_closeness[j] = closeness_j
#     return user_closeness

# # ====================== PLOT FULL LINE + KEY EVENTS ======================

# def plot_dtw_closeness_with_key_events(
#     bench_times, bench_events,
#     user_times_offset, user_events_offset,
#     user_closeness,
#     path,
#     title="DTW Closeness (All Frames) + Key Events"
# ):
#     """
#     1) Plot the full closeness line: benchmark pinned at 100, user_closeness in [0..100]
#        across the entire time range.
#     2) Only highlight the key events (Start, WristAboveShoulder, etc.) as big markers.
#     3) Connect the matching events with dotted lines.
#     4) We shift user_times by the offset so both 'Start' events line up in time.
#     """
#     fig, ax = plt.subplots(figsize=(10, 5))

#     # Benchmark line (all frames)
#     # We'll create a simple x for the benchmark = 0..N-1 => we can also do bench_times
#     # But to show a "line" for the benchmark, we can do:
#     bench_y = np.full_like(bench_times, 100)  # pinned at 100
#     ax.plot(bench_times, bench_y, '-', color='orange', label="Benchmark (100%)")

#     # User line (all frames)
#     ax.plot(user_times_offset, user_closeness, '-', color='blue', label="User Closeness (%)")

#     # Dotted lines for every DTW pair
#     # (Optional: if you prefer not to see them all, comment this out)
#     for (i, j) in path:
#         x_b = bench_times[i]
#         y_b = 100
#         x_u = user_times_offset[j]
#         y_u = user_closeness[j]
#         ax.plot([x_b, x_u], [y_b, y_u], 'k--', alpha=0.1)

#     # Mark key events
#     # Benchmark events are pinned at y=100
#     for (evt_name, evt_t) in bench_events:
#         # find nearest index in bench_times
#         i_closest = (np.abs(bench_times - evt_t)).argmin()
#         x_b = bench_times[i_closest]
#         y_b = 100
#         ax.scatter(x_b, y_b, marker='x', color='orange', s=100)
#         ax.text(x_b, y_b+3, evt_name, color='orange', ha='center', va='bottom', fontsize=9)

#     # User events => scatter at user_closeness
#     for (evt_name, evt_t) in user_events_offset:
#         # find nearest index in user_times_offset
#         # user_times_offset is same length as user_times
#         j_closest = (np.abs(user_times_offset - evt_t)).argmin()
#         x_u = user_times_offset[j_closest]
#         y_u = user_closeness[j_closest]
#         ax.scatter(x_u, y_u, marker='o', color='blue', s=70)
#         ax.text(x_u, y_u-3, evt_name, color='blue', ha='center', va='top', fontsize=9)

#         # Optionally connect matching events with dotted lines
#         # We can find which i matched j, but that might not be the same event
#         # If you want to forcibly connect them, do so here
#         # or if you want to connect event->event, you need i from bench_events and j from user_events
#         # for the same event_name. This is optional.

#     ax.set_title(title)
#     ax.set_xlabel("Time (seconds)")
#     ax.set_ylabel("Closeness to Benchmark (%)")
#     ax.set_ylim([0, 110])
#     ax.grid(True)
#     ax.legend()
#     plt.show()

# # ====================== MAIN SCRIPT ======================

# def main():
#     print("First, record a 'benchmark' shot (the test shot).")
#     benchmark_data = record_shot(prompt="Benchmark Shot")

#     print("Now record a 'user' shot to compare.")
#     user_data = record_shot(prompt="User Shot")

#     # 1) Extract form
#     bench_times, bench_form = extract_form_series(benchmark_data)
#     user_times,  user_form  = extract_form_series(user_data)

#     # 2) Suppose we define events by their time in seconds for each shot
#     #    (In real code, you'd detect them automatically or define them manually.)
#     #    We'll store them as: [("Start", timeVal), ("RightWristAboveShoulder", timeVal), ...]
#     # For demonstration, let's assume the first frame is "Start", frame ~ middle is "WristAboveShoulder", etc.
#     # We'll do a quick guess:
#     bench_events = [
#         ("Start", bench_times[0]),
#         ("RightWristAboveShoulder", bench_times[len(bench_times)//3]),
#         ("RightElbowAboveShoulder", bench_times[len(bench_times)//2]),
#         ("FollowThrough", bench_times[-1])
#     ]
#     user_events = [
#         ("Start", user_times[0]),
#         ("RightWristAboveShoulder", user_times[len(user_times)//3]),
#         ("RightElbowAboveShoulder", user_times[len(user_times)//2]),
#         ("FollowThrough", user_times[-1])
#     ]

#     # 3) Align the user so that user "Start" is at the same time as bench "Start"
#     user_times_offset = align_start_event(bench_times, user_times, bench_events, user_events, start_event_name="Start")

#     # Also shift user_events by the same offset
#     offset = (bench_events[0][1] - user_events[0][1])
#     user_events_offset = []
#     for (evt_name, evt_t) in user_events:
#         user_events_offset.append((evt_name, evt_t + offset))

#     # 4) Run DTW
#     bench_reshaped = bench_form.reshape(-1, 1)
#     user_reshaped  = user_form.reshape(-1, 1)
#     distance, path = fastdtw(bench_reshaped, user_reshaped, dist=euclidean)
#     print(f"DTW distance: {distance:.2f}")

#     # 5) Compute user closeness
#     user_closeness = compute_user_closeness(bench_form, user_form, path)

#     # 6) Plot the entire closeness line + key events
#     plot_dtw_closeness_with_key_events(
#         bench_times,
#         bench_events,
#         user_times_offset,
#         user_events_offset,
#         user_closeness,
#         path,
#         title="DTW Closeness (All Frames) + Key Events (Aligned Start)"
#     )

# if __name__ == "__main__":
#     main()

# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import matplotlib
# import matplotlib.pyplot as plt

# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean

# # ====================== CONFIG / CONSTANTS ======================

# NEUTRAL_COLOR = (0, 255, 0)
# PRE_SHOT_COLOR = (0, 165, 255)
# FOLLOW_THROUGH_COLOR = (255, 255, 0)

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# RIGHT_LANDMARKS = [12, 14, 16]
# LEFT_LANDMARKS  = [11, 13, 15]

# pose = mp_pose.Pose(
#     model_complexity=2,
#     static_image_mode=False,
#     smooth_landmarks=True,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# # Global references to store frames for overlay
# landmark_frames_bench = []
# landmark_frames_user  = []

# # ====================== POSE & ANGLE UTILS ======================

# def get_3d_point(landmarks, index, width, height):
#     if index >= len(landmarks) or landmarks[index].visibility < 0.5:
#         return None
#     return np.array([
#         landmarks[index].x * width,
#         landmarks[index].y * height,
#         landmarks[index].z
#     ])

# def calculate_3d_angle(a, b, c):
#     if a is None or b is None or c is None:
#         return None
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
#     if denom < 1e-5:
#         return None
#     cosine_angle = np.dot(ba, bc) / denom
#     return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# def get_arm_state(landmarks, width, height):
#     right_shoulder = get_3d_point(landmarks, 12, width, height)
#     right_elbow    = get_3d_point(landmarks, 14, width, height)
#     right_wrist    = get_3d_point(landmarks, 16, width, height)
#     left_wrist     = get_3d_point(landmarks, 15, width, height)
#     left_hip       = get_3d_point(landmarks, 23, width, height)
#     right_hip      = get_3d_point(landmarks, 24, width, height)

#     if (right_wrist is not None and left_wrist is not None and
#         left_hip is not None and right_hip is not None and
#         right_shoulder is not None):
#         waist_y = (left_hip[1] + right_hip[1]) / 2.0
#         avg_wrist_y = (right_wrist[1] + left_wrist[1]) / 2.0
#         dist_wrists = np.linalg.norm(right_wrist - left_wrist)
#         if (dist_wrists < 0.15 * width and avg_wrist_y < waist_y
#             and right_wrist[1] > right_shoulder[1]):
#             return "pre_shot"

#     if right_wrist is not None and right_shoulder is not None:
#         if right_shoulder[1] > right_wrist[1]:
#             return "follow_through"

#     if right_shoulder is not None and right_wrist is not None:
#         if right_wrist[1] > right_shoulder[1]:
#             return "neutral"

#     return "neutral"

# # ====================== CAPTURE & STORE SHOT ANGLES + FRAMES ======================

# def record_shot(prompt="Recording Shot"):
#     """
#     Returns:
#       shot_angles: list of (state, time, elbow_angle, wrist_angle, arm_angle)
#       landmark_frames: list of (time, 33x3 array) for replay
#     """
#     cap = cv2.VideoCapture(0)
#     print(f"\n--- {prompt}: Please perform your shot. Press 'q' to quit early. ---")
#     shot_angles = []
#     landmark_frames = []  # store full body landmarks each frame

#     previous_stage = "neutral"
#     start_time = None
#     last_print_time = time.time()
#     done = False

#     while not done:
#         ret, frame = cap.read()
#         if not ret:
#             print("No frame captured. Exiting.")
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)
        
#         h, w, _ = frame.shape
#         current_time = time.time()

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             state = get_arm_state(landmarks, w, h)

#             # Draw
#             if state == "pre_shot":
#                 current_color = PRE_SHOT_COLOR
#                 status_text   = "Pre-Shot"
#             elif state == "follow_through":
#                 current_color = FOLLOW_THROUGH_COLOR
#                 status_text   = "Follow Through"
#             else:
#                 current_color = NEUTRAL_COLOR
#                 status_text   = "Neutral"
            
#             cv2.putText(frame, status_text, (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#             mp_drawing.draw_landmarks(
#                 frame,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(
#                     color=current_color, thickness=2, circle_radius=3
#                 ),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(
#                     color=current_color, thickness=2
#                 )
#             )

#             # Compute angles
#             right_shoulder = get_3d_point(landmarks, 12, w, h)
#             right_elbow    = get_3d_point(landmarks, 14, w, h)
#             right_wrist    = get_3d_point(landmarks, 16, w, h)

#             elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
#             wrist_angle = calculate_3d_angle(right_elbow, right_wrist,
#                                              get_3d_point(landmarks, 20, w, h))
#             arm_angle   = calculate_3d_angle(get_3d_point(landmarks, 11, w, h),
#                                              right_shoulder, right_elbow)

#             # If in pre_shot/follow_through, record angles
#             if state in ["pre_shot", "follow_through"]:
#                 if start_time is None:
#                     start_time = current_time
#                 if current_time - last_print_time >= 0.1:
#                     elapsed_time = (current_time - start_time)
#                     shot_angles.append((state, elapsed_time,
#                                         elbow_angle, wrist_angle, arm_angle))
#                     last_print_time = current_time

#             # Store full body landmarks for replay
#             # shape: (33,3)
#             frame_landmarks_3d = np.full((33,3), np.nan, dtype=np.float32)
#             for i in range(33):
#                 p = get_3d_point(landmarks, i, w, h)
#                 if p is not None:
#                     frame_landmarks_3d[i] = p
#             # store (time, 33x3 array)
#             if start_time is not None:
#                 shot_time = current_time - start_time
#             else:
#                 shot_time = 0.0
#             landmark_frames.append((shot_time, frame_landmarks_3d))

#             # Stage transitions
#             if state != previous_stage:
#                 if state == "pre_shot":
#                     start_time = current_time
#                     shot_angles = []
#                     landmark_frames = []
#                     last_print_time = start_time
#                 elif state == "neutral":
#                     if shot_angles:
#                         phases = [x[0] for x in shot_angles]
#                         if "pre_shot" in phases and "follow_through" in phases:
#                             print("\nShot Completed.")
#                             done = True
#                 previous_stage = state

#         cv2.imshow(prompt, frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Quit early.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return shot_angles, landmark_frames

# # ====================== OVERALL FORM, DTW, CLOSENESS ======================

# def compute_overall_form(e, w, a):
#     angles = []
#     if e is not None:
#         angles.append(e)
#     if w is not None:
#         angles.append(w)
#     if a is not None:
#         angles.append(a)
#     if len(angles) == 0:
#         return None
#     return sum(angles) / len(angles)

# def extract_form_series(shot_data):
#     times = []
#     form_vals = []
#     for entry in shot_data:
#         t  = entry[1]
#         e  = entry[2]
#         w  = entry[3]
#         a  = entry[4]
#         measure = compute_overall_form(e, w, a)
#         if measure is not None:
#             times.append(t)
#             form_vals.append(measure)
#     return np.array(times), np.array(form_vals)

# def compute_user_closeness(bench_form, user_form, path):
#     alpha = 2.0
#     user_map = {}
#     for (i, j) in path:
#         if j not in user_map:
#             user_map[j] = []
#         user_map[j].append(i)

#     user_closeness = np.zeros_like(user_form)
#     for j in range(len(user_form)):
#         if j in user_map:
#             i_list = user_map[j]
#             i_mid = i_list[len(i_list)//2]
#             diff = abs(user_form[j] - bench_form[i_mid])
#             closeness_j = 100 - alpha * diff
#             closeness_j = max(0, min(100, closeness_j))
#         else:
#             closeness_j = 100
#         user_closeness[j] = closeness_j
#     return user_closeness

# # ====================== SHIFT TIME FOR "START" ALIGNMENT ======================

# def align_start(bench_times, user_times, start_name="Start"):
#     # optional if you want to find actual "Start" time in each array
#     # for simplicity, let's just shift so user_times[0] = bench_times[0]
#     offset = bench_times[0] - user_times[0]
#     return user_times + offset

# # ====================== 3D OVERLAY PLACEHOLDER ======================

# def show_overlay_window(bench_time, user_time,
#                         landmark_frames_bench, landmark_frames_user):
#     """
#     Displays a side-by-side overlay from (bench_time-0.25..bench_time+0.25) and
#     (user_time-0.25..user_time+0.25). We'll do a simple 2D "skeleton" approach.

#     NOTE: For a 3D animation, you'd adapt your older "animate_pose_sequence" code here.
#     """
#     dt = 0.25
#     bench_start = bench_time - dt
#     bench_end   = bench_time + dt
#     user_start  = user_time  - dt
#     user_end    = user_time  + dt

#     print(f"Showing overlay from {bench_start:.2f}..{bench_end:.2f} (benchmark), "
#           f"{user_start:.2f}..{user_end:.2f} (user).")

#     # 1) Filter frames for the given time window
#     bench_subframes = [(t, f) for (t, f) in landmark_frames_bench
#                        if (t >= bench_start and t <= bench_end)]
#     user_subframes  = [(t, f) for (t, f) in landmark_frames_user
#                        if (t >= user_start  and t <= user_end)]

#     if not bench_subframes or not user_subframes:
#         print("No frames found in that time range.")
#         return

#     # 2) Create a new figure with side-by-side subplots
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))
#     ax_bench, ax_user = axes

#     ax_bench.set_title("Benchmark Overlay")
#     ax_bench.set_xlim([0, 640])  # or adjust to your typical x-range
#     ax_bench.set_ylim([480, 0])  # flip Y if you want top-down
#     ax_bench.invert_yaxis()      # or not, depending on your coordinate system

#     ax_user.set_title("User Overlay")
#     ax_user.set_xlim([0, 640])
#     ax_user.set_ylim([480, 0])
#     ax_user.invert_yaxis()

#     # 3) Draw skeleton for each frame in that time window
#     #    We'll do a quick approach: scatter the 33 landmarks + connect with POSE_CONNECTIONS
#     import time

#     for (t_bench, frame_bench) in bench_subframes:
#         # Let's pick a color or alpha based on time
#         alpha = 0.5
#         draw_2d_skeleton(ax_bench, frame_bench, alpha=alpha)

#     for (t_user, frame_user) in user_subframes:
#         alpha = 0.5
#         draw_2d_skeleton(ax_user, frame_user, alpha=alpha)

#     plt.tight_layout()
#     plt.show()


# def draw_2d_skeleton(ax, landmarks_3d, alpha=1.0):
#     """
#     Minimal 2D skeleton draw: We have 33x3 array. We'll treat x=..., y=...
#     ignoring z. Then connect POSE_CONNECTIONS in 2D.

#     NOTE: This is purely a placeholder. You can adapt to your coordinate system.
#     """
#     # landmarks_3d is shape (33,3). We treat columns [0,1] as x,y
#     import numpy as np

#     for (start_idx, end_idx) in POSE_CONNECTIONS:
#         p1 = landmarks_3d[start_idx]
#         p2 = landmarks_3d[end_idx]
#         # check for NaNs
#         if not np.isnan(p1[0]) and not np.isnan(p2[0]):
#             ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
#                     color='red', alpha=alpha, linewidth=2)

#     # Optionally scatter the points
#     xs = landmarks_3d[:, 0]
#     ys = landmarks_3d[:, 1]
#     valid_mask = ~np.isnan(xs) & ~np.isnan(ys)
#     ax.scatter(xs[valid_mask], ys[valid_mask],
#                c='blue', alpha=alpha, s=20)

# # ====================== PICK HANDLER ======================

# def onpick(event):
#     artist = event.artist
#     if not hasattr(artist, 'my_label'):
#         return

#     label = artist.my_label
#     ind = event.ind[0]  # index of the picked point

#     if label == "bench_events":
#         # We stored bench_event_times, bench_event_names
#         # The user clicked on the benchmark event point
#         # We can find the time and name from arrays
#         bench_time = bench_event_times[ind]
#         evt_name   = bench_event_names[ind]
#         # Suppose we find the matched user time
#         user_time  = user_event_times[ind]  # or some logic
#         print(f"Clicked on benchmark event {evt_name} at time {bench_time:.2f}")
#         show_overlay_window(bench_time, user_time,
#                             landmark_frames_bench, landmark_frames_user)

#     elif label == "user_events":
#         # Similarly
#         user_time = user_event_times[ind]
#         evt_name  = user_event_names[ind]
#         bench_time = bench_event_times[ind]
#         print(f"Clicked on user event {evt_name} at time {user_time:.2f}")
#         show_overlay_window(bench_time, user_time,
#                             landmark_frames_bench, landmark_frames_user)

# # ====================== PLOT FUNCTION ======================

# def plot_interactive_closeness(bench_times, user_times, user_closeness,
#                                bench_events, user_events):
#     """
#     Plot a continuous line for user_closeness (blue) vs. user_times,
#     benchmark pinned at 100 (orange) vs. bench_times,
#     plus key events as clickable scatter points with no text.
#     """
#     fig, ax = plt.subplots(figsize=(10, 5))

#     # Benchmark line
#     bench_y = np.full_like(bench_times, 100)
#     ax.plot(bench_times, bench_y, color='orange', label="Benchmark (100%)")

#     # User closeness line
#     ax.plot(user_times, user_closeness, color='blue', label="User Closeness (%)")

#     # Scatter for benchmark events
#     # We'll store them in arrays so we can do "picker=True"
#     global bench_event_times, bench_event_names
#     bench_event_times = []
#     bench_event_names = []
#     for (evt_name, evt_t) in bench_events:
#         bench_event_times.append(evt_t)
#         bench_event_names.append(evt_name)

#     bench_event_times = np.array(bench_event_times)
#     bench_scatter = ax.scatter(bench_event_times, np.full(len(bench_event_times), 100),
#                                c='orange', marker='x', picker=True)
#     bench_scatter.my_label = "bench_events"

#     # Scatter for user events
#     global user_event_times, user_event_names
#     user_event_times = []
#     user_event_names = []
#     for (evt_name, evt_t) in user_events:
#         user_event_times.append(evt_t)
#         user_event_names.append(evt_name)

#     user_event_times = np.array(user_event_times)
#     user_scatter = ax.scatter(user_event_times,
#                               [user_closeness[(np.abs(user_times - t)).argmin()]
#                                for t in user_event_times],
#                               c='blue', marker='o', picker=True)
#     user_scatter.my_label = "user_events"

#     fig.canvas.mpl_connect('pick_event', onpick)

#     ax.set_title("DTW Closeness Over Time (Interactive Key Events)")
#     ax.set_xlabel("Time (seconds)")
#     ax.set_ylabel("Closeness to Benchmark (%)")
#     ax.set_ylim([0, 110])
#     ax.grid(True)
#     ax.legend()
#     plt.show()

# # ====================== MAIN SCRIPT ======================

# def main():
#     print("Recording benchmark shot...")
#     benchmark_data, frames_bench = record_shot(prompt="Benchmark Shot")
#     print("Recording user shot...")
#     user_data, frames_user = record_shot(prompt="User Shot")

#     # Save them globally so "show_overlay_window" can access
#     global landmark_frames_bench, landmark_frames_user
#     landmark_frames_bench = frames_bench
#     landmark_frames_user  = frames_user

#     # Extract form
#     bench_times, bench_form = extract_form_series(benchmark_data)
#     user_times,  user_form  = extract_form_series(user_data)

#     # Align user start if you want
#     offset = bench_times[0] - user_times[0]
#     user_times_aligned = user_times + offset

#     # Run DTW
#     distance, path = fastdtw(bench_form.reshape(-1,1), user_form.reshape(-1,1), dist=euclidean)
#     print(f"DTW distance: {distance:.2f}")

#     # Compute closeness
#     user_closeness = compute_user_closeness(bench_form, user_form, path)

#     # Suppose we define key events by time in each shot
#     # In reality, you'd do something more precise
#     bench_events = [
#         ("Start", bench_times[0]),
#         ("WristAboveShoulder", bench_times[len(bench_times)//3]),
#         ("ElbowAboveShoulder", bench_times[len(bench_times)//2]),
#         ("FollowThrough", bench_times[-1])
#     ]
#     user_events = [
#         ("Start", user_times_aligned[0]),
#         ("WristAboveShoulder", user_times_aligned[len(user_times_aligned)//3]),
#         ("ElbowAboveShoulder", user_times_aligned[len(user_times_aligned)//2]),
#         ("FollowThrough", user_times_aligned[-1])
#     ]

#     # Plot interactive closeness
#     plot_interactive_closeness(bench_times, user_times_aligned,
#                                user_closeness, bench_events, user_events)

# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ====================== CONFIG / CONSTANTS ======================

NEUTRAL_COLOR = (0, 255, 0)
PRE_SHOT_COLOR = (0, 165, 255)
FOLLOW_THROUGH_COLOR = (255, 255, 0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

RIGHT_LANDMARKS = [12, 14, 16]
LEFT_LANDMARKS  = [11, 13, 15]

pose = mp_pose.Pose(
    model_complexity=2,
    static_image_mode=False,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# We'll store the final frames for benchmark/user if needed globally
landmark_frames_bench = []
landmark_frames_user  = []

# ====================== POSE & ANGLE UTILS ======================

def get_3d_point(landmarks, index, width, height):
    """Extract (x, y, z) from Mediapipe landmarks. Returns None if not visible enough."""
    if index >= len(landmarks) or landmarks[index].visibility < 0.5:
        return None
    return np.array([
        landmarks[index].x * width,
        landmarks[index].y * height,
        landmarks[index].z
    ])

def calculate_3d_angle(a, b, c):
    """Compute angle at b formed by points a->b->c in 3D."""
    if a is None or b is None or c is None:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-5:
        return None
    cosine_angle = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_arm_state(landmarks, width, height):
    """
    'pre_shot' => wrists close together below shoulders
    'follow_through' => wrist above shoulder
    'neutral' => default
    """
    right_shoulder = get_3d_point(landmarks, 12, width, height)
    right_elbow    = get_3d_point(landmarks, 14, width, height)
    right_wrist    = get_3d_point(landmarks, 16, width, height)
    left_wrist     = get_3d_point(landmarks, 15, width, height)
    left_hip       = get_3d_point(landmarks, 23, width, height)
    right_hip      = get_3d_point(landmarks, 24, width, height)

    if (right_wrist is not None and left_wrist is not None and
        left_hip is not None and right_hip is not None and
        right_shoulder is not None):
        waist_y = (left_hip[1] + right_hip[1]) / 2.0
        avg_wrist_y = (right_wrist[1] + left_wrist[1]) / 2.0
        dist_wrists = np.linalg.norm(right_wrist - left_wrist)
        if (dist_wrists < 0.15 * width and avg_wrist_y < waist_y
            and right_wrist[1] > right_shoulder[1]):
            return "pre_shot"

    if right_wrist is not None and right_shoulder is not None:
        if right_shoulder[1] > right_wrist[1]:
            return "follow_through"

    if right_shoulder is not None and right_wrist is not None:
        if right_wrist[1] > right_shoulder[1]:
            return "neutral"

    return "neutral"

# ====================== CAPTURE & STORE SHOT ANGLES + FRAMES ======================

def record_shot(prompt="Recording Shot"):
    """
    Records a single shot. Returns:
      shot_angles: list of (state, time, elbow_angle, wrist_angle, arm_angle)
      landmark_frames: list of (time, 33x3 array)
    """
    cap = cv2.VideoCapture(0)
    print(f"\n--- {prompt}: Please perform your shot. Press 'q' to quit early. ---")
    shot_angles = []
    landmark_frames = []

    previous_stage = "neutral"
    start_time = None
    last_print_time = time.time()
    done = False
    recording_active = False
    seen_follow_through = False

    while not done:
        ret, frame = cap.read()
        if not ret:
            print("No frame captured. Exiting.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        h, w, _ = frame.shape
        current_time = time.time()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            state = get_arm_state(landmarks, w, h)

            # Decide color & text
            if state == "pre_shot":
                current_color = PRE_SHOT_COLOR
                status_text   = "Pre-Shot"
            elif state == "follow_through":
                current_color = FOLLOW_THROUGH_COLOR
                status_text   = "Follow Through"
            else:
                current_color = NEUTRAL_COLOR
                status_text   = "Neutral"
            
            cv2.putText(frame, status_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=current_color, thickness=2, circle_radius=3
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=current_color, thickness=2
                )
            )

            # Compute angles
            right_shoulder = get_3d_point(landmarks, 12, w, h)
            right_elbow    = get_3d_point(landmarks, 14, w, h)
            right_wrist    = get_3d_point(landmarks, 16, w, h)

            elbow_angle = calculate_3d_angle(right_shoulder, right_elbow, right_wrist)
            wrist_angle = calculate_3d_angle(right_elbow, right_wrist,
                                             get_3d_point(landmarks, 20, w, h))
            arm_angle   = calculate_3d_angle(get_3d_point(landmarks, 11, w, h),
                                             right_shoulder, right_elbow)

            # Store full body landmarks for replay
            frame_landmarks_3d = np.full((33,3), np.nan, dtype=np.float32)
            for i in range(33):
                p = get_3d_point(landmarks, i, w, h)
                if p is not None:
                    frame_landmarks_3d[i] = p

            # Stage transitions
            if state != previous_stage:
                # Start recording when entering pre_shot
                if state == "pre_shot" and not recording_active:
                    recording_active = True
                    seen_follow_through = False
                    start_time = current_time
                    shot_angles = []
                    landmark_frames = []
                    last_print_time = start_time

                # Reset if pre_shot is followed by neutral (before follow_through)
                elif state == "neutral" and recording_active and not seen_follow_through:
                    recording_active = False
                    seen_follow_through = False
                    start_time = None
                    shot_angles = []
                    landmark_frames = []

                # Mark follow_through reached
                elif state == "follow_through" and recording_active:
                    seen_follow_through = True

                # Complete when we see pre_shot after follow_through
                elif state == "pre_shot" and recording_active and seen_follow_through:
                    # Record this final frame
                    elapsed = current_time - start_time
                    landmark_frames.append((elapsed, frame_landmarks_3d))
                    print("\nShot Completed.")
                    done = True

                previous_stage = state

            # Record frames while actively recording
            if recording_active and not done:
                elapsed = current_time - start_time
                landmark_frames.append((elapsed, frame_landmarks_3d))
                
                # Store angles for pre_shot/follow_through states
                if state in ["pre_shot", "follow_through"]:
                    if current_time - last_print_time >= 0.1:
                        shot_angles.append((state, elapsed,
                                            elbow_angle, wrist_angle, arm_angle))
                        last_print_time = current_time

        cv2.imshow(prompt, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return shot_angles, landmark_frames

# ====================== OPTIONAL ANGLE/DTW CODE (unused) ======================

def compute_overall_form(e, w, a):
    angles = []
    if e is not None:
        angles.append(e)
    if w is not None:
        angles.append(w)
    if a is not None:
        angles.append(a)
    if len(angles) == 0:
        return None
    return sum(angles) / len(angles)

def extract_form_series(shot_data):
    times = []
    form_vals = []
    for entry in shot_data:
        t  = entry[1]
        e  = entry[2]
        w  = entry[3]
        a  = entry[4]
        measure = compute_overall_form(e, w, a)
        if measure is not None:
            times.append(t)
            form_vals.append(measure)
    return np.array(times), np.array(form_vals)

def compute_user_closeness(bench_form, user_form, path):
    # not used for ghost overlay
    return None

def generate_feedback(benchmark_data, user_data, bench_times, user_times, user_closeness):
    """
    Generate actionable feedback comparing user shot to benchmark.
    Returns a list of feedback strings.
    """
    feedback = []
    
    # 1. Overall score
    avg_closeness = np.mean(user_closeness) if len(user_closeness) > 0 else 0
    feedback.append(f"\n=== OVERALL SCORE: {avg_closeness:.1f}% ===")
    
    if avg_closeness >= 90:
        feedback.append("Excellent form! Your shot closely matches the benchmark.")
    elif avg_closeness >= 75:
        feedback.append("Good form with room for improvement.")
    elif avg_closeness >= 60:
        feedback.append("Your form needs work. Focus on key areas below.")
    else:
        feedback.append("Significant differences detected. Review the specific feedback below.")
    
    # 2. Timing comparison
    bench_duration = bench_times[-1] - bench_times[0] if len(bench_times) > 1 else 0
    user_duration = user_times[-1] - user_times[0] if len(user_times) > 1 else 0
    if bench_duration > 0 and user_duration > 0:
        time_diff_pct = ((user_duration - bench_duration) / bench_duration) * 100
        if abs(time_diff_pct) > 10:
            if time_diff_pct > 0:
                feedback.append(f"⏱️  TIMING: Your shot is {time_diff_pct:.1f}% slower than the benchmark. Try to maintain a quicker, more fluid motion.")
            else:
                feedback.append(f"⏱️  TIMING: Your shot is {abs(time_diff_pct):.1f}% faster than the benchmark. Consider slowing down slightly for better control.")
    
    # 3. Key event comparison
    def get_event_angles(shot_data, times, event_idx):
        """Get angles at a specific event (by index: 0=start, 1=middle, 2=release, 3=end)"""
        if len(times) == 0 or len(shot_data) == 0:
            return None, None, None
        
        if event_idx == 0:
            target_idx = 0
        elif event_idx == 1:
            target_idx = len(times) // 3
        elif event_idx == 2:
            target_idx = (2 * len(times)) // 3
        else:
            target_idx = len(times) - 1
        
        if target_idx < len(shot_data):
            entry = shot_data[target_idx]
            return entry[2], entry[3], entry[4]  # elbow, wrist, arm
        return None, None, None
    
    event_names = ["Start", "Ball Set", "Release", "Follow Through"]
    event_issues = []
    
    for i, event_name in enumerate(event_names):
        bench_elbow, bench_wrist, bench_arm = get_event_angles(benchmark_data, bench_times, i)
        user_elbow, user_wrist, user_arm = get_event_angles(user_data, user_times, i)
        
        if bench_elbow is not None and user_elbow is not None:
            elbow_diff = abs(user_elbow - bench_elbow)
            wrist_diff = abs(user_wrist - bench_wrist) if user_wrist is not None and bench_wrist is not None else 0
            arm_diff = abs(user_arm - bench_arm) if user_arm is not None and bench_arm is not None else 0
            
            # Basketball-specific actionable feedback
            if event_name == "Follow Through":
                # Wrist snap analysis at follow-through
                if user_wrist is not None and bench_wrist is not None and wrist_diff > 10:
                    if user_wrist > bench_wrist + 5:
                        event_issues.append(f"💪 {event_name}: Your wrist isn't snapping hard enough. Actively snap your wrist forward at release for better follow-through and shot power.")
                    elif user_wrist < bench_wrist - 5:
                        event_issues.append(f"💪 {event_name}: Your wrist is over-extending. Focus on a controlled snap - not too hard, not too soft.")
                
                # Elbow extension at follow-through
                if elbow_diff > 10:
                    if user_elbow > bench_elbow + 5:
                        event_issues.append(f"💪 {event_name}: Keep your arm fully extended after release. Don't let your elbow collapse - maintain full extension for better arc.")
                    elif user_elbow < bench_elbow - 5:
                        event_issues.append(f"💪 {event_name}: Your arm is too straight. Maintain a slight natural bend even at full extension.")
            
            elif event_name == "Release":
                # Wrist position at release
                if user_wrist is not None and bench_wrist is not None and wrist_diff > 10:
                    if user_wrist > bench_wrist + 5:
                        event_issues.append(f"🎯 {event_name}: Your wrist is too bent at release. Snap your wrist forward more aggressively - this creates the backspin for better accuracy.")
                    elif user_wrist < bench_wrist - 5:
                        event_issues.append(f"🎯 {event_name}: Release with your wrist in a more bent position, then snap forward. This creates the proper shooting motion.")
                
                # Elbow position at release
                if elbow_diff > 15:
                    if user_elbow > bench_elbow + 10:
                        event_issues.append(f"🎯 {event_name}: Your elbow is too wide (chicken wing). Keep your elbow closer to your body and aligned with the rim.")
                    elif user_elbow < bench_elbow - 10:
                        event_issues.append(f"🎯 {event_name}: Your elbow is too tucked in. Find the balance - not too wide, not too tight.")
            
            elif event_name == "Ball Set":
                # Shooting pocket position
                if elbow_diff > 15:
                    if user_elbow > bench_elbow + 10:
                        event_issues.append(f"🏀 {event_name}: Your elbow is flaring out. Keep your shooting arm aligned - elbow should point toward the rim, not outward.")
                    else:
                        event_issues.append(f"🏀 {event_name}: Tuck your elbow in slightly more. Your shooting arm should form a smooth L-shape.")
                
                # Wrist preparation
                if user_wrist is not None and bench_wrist is not None and wrist_diff > 15:
                    event_issues.append(f"🏀 {event_name}: Prepare your wrist for the shot. Your wrist should be cocked back and ready to snap forward.")
            
            elif event_name == "Start":
                # Initial setup
                if elbow_diff > 15:
                    if user_elbow > bench_elbow + 10:
                        event_issues.append(f"🏀 {event_name}: Start with your elbow closer to your body. Your shooting arm should be relaxed but ready.")
                    else:
                        event_issues.append(f"🏀 {event_name}: Start with your elbow slightly more bent. Prepare your shooting pocket early.")
                
                # Overall form check
                if arm_diff > 15 and user_arm is not None:
                    event_issues.append(f"🏀 {event_name}: Check your shooting stance. Your shooting arm should be aligned with your target from the start.")
    
    feedback.extend(event_issues[:4])  # Show top 4 issues
    
    # 4. Identify worst phase
    if len(user_closeness) > 0:
        min_closeness_idx = np.argmin(user_closeness)
        min_closeness = user_closeness[min_closeness_idx]
        if min_closeness < 70:
            phase_ratio = min_closeness_idx / len(user_closeness) if len(user_closeness) > 0 else 0
            if phase_ratio < 0.33:
                phase_name = "Start/Ball Set"
            elif phase_ratio < 0.67:
                phase_name = "Release"
            else:
                phase_name = "Follow Through"
            feedback.append(f"⚠️  WORST PHASE: {phase_name} ({min_closeness:.1f}% match) - Focus on improving this phase.")
    
    # 5. Additional basketball-specific tips
    # Check for consistency issues
    if len(user_closeness) > 0:
        closeness_std = np.std(user_closeness)
        if closeness_std > 15:
            feedback.append("⚠️  CONSISTENCY: Your form varies throughout the shot. Focus on maintaining consistent mechanics from start to finish.")
    
    # Check for wrist snap consistency (compare release to follow-through)
    if len(user_data) > 0 and len(benchmark_data) > 0:
        # Get wrist angles at release and follow-through
        release_wrist = get_event_angles(user_data, user_times, 2)[1]  # wrist at release
        follow_through_wrist = get_event_angles(user_data, user_times, 3)[1]  # wrist at follow-through
        bench_release_wrist = get_event_angles(benchmark_data, bench_times, 2)[1]
        bench_follow_through_wrist = get_event_angles(benchmark_data, bench_times, 3)[1]
        
        if (release_wrist is not None and follow_through_wrist is not None and 
            bench_release_wrist is not None and bench_follow_through_wrist is not None):
            # Wrist should snap forward (wrist angle should decrease from release to follow-through)
            user_wrist_snap = release_wrist - follow_through_wrist
            bench_wrist_snap = bench_release_wrist - bench_follow_through_wrist
            snap_diff = user_wrist_snap - bench_wrist_snap
            
            if snap_diff < -5:
                feedback.append("💪 WRIST SNAP: You're not snapping your wrist aggressively enough. The snap creates backspin - practice the 'gooseneck' follow-through.")
            elif snap_diff > 10:
                feedback.append("💪 WRIST SNAP: Your wrist snap is too aggressive. Aim for a controlled, smooth snap, not a violent motion.")
    
    # 6. Positive reinforcement
    if avg_closeness >= 75:
        max_closeness = np.max(user_closeness) if len(user_closeness) > 0 else 0
        if max_closeness > 95:
            feedback.append("✅ Great job! Some phases match the benchmark perfectly.")
        if avg_closeness >= 85:
            feedback.append("💡 TIP: Your form is solid! Focus on repetition to build muscle memory.")
    
    return feedback

# ====================== GHOST OVERLAY (BENCH + USER ON SAME AXES) ======================

def side_angle_coords(landmarks_3d, mode='xy'):
    """
    Convert 3D coords -> 2D for a side angle. If mode='xy', we do (x,y).
    If mode='zy', we do (z,y).
    """
    coords_2d = []
    for (x, y, z) in landmarks_3d:
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            coords_2d.append([np.nan, np.nan])
            continue
        if mode == 'xy':
            coords_2d.append([x, y])
        else:
            coords_2d.append([z, y])  # fallback
    return np.array(coords_2d, dtype=np.float32)

def draw_2d_skeleton(ax, coords_2d, alpha=0.5, color='blue'):
    import math
    for (start_idx, end_idx) in POSE_CONNECTIONS:
        x1, y1 = coords_2d[start_idx]
        x2, y2 = coords_2d[end_idx]
        if (not math.isnan(x1)) and (not math.isnan(x2)):
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=2)
    # scatter
    valid_mask = ~np.isnan(coords_2d[:,0]) & ~np.isnan(coords_2d[:,1])
    ax.scatter(coords_2d[valid_mask,0], coords_2d[valid_mask,1],
               color=color, alpha=alpha, s=20)

def _compute_axis_limits(bench_frames, user_frames, mode='xy'):
    """
    Compute dynamic axis limits from recorded landmark coordinates to ensure
    skeletons are visible regardless of camera resolution.
    """
    all_x = []
    all_y = []
    for (_, arr) in bench_frames + user_frames:
        if arr is None or len(arr) == 0:
            continue
        if mode == 'xy':
            xs = arr[:, 0]
            ys = arr[:, 1]
        else:
            xs = arr[:, 2]
            ys = arr[:, 1]
        mask = (~np.isnan(xs)) & (~np.isnan(ys))
        if np.any(mask):
            all_x.append(xs[mask])
            all_y.append(ys[mask])
    if not all_x:
        # sensible defaults (typical 640x480)
        return (0, 640), (0, 480)
    xs = np.concatenate(all_x)
    ys = np.concatenate(all_y)
    # add small padding
    pad_x = max(10.0, 0.05 * (np.nanmax(xs) - np.nanmin(xs) + 1e-3))
    pad_y = max(10.0, 0.05 * (np.nanmax(ys) - np.nanmin(ys) + 1e-3))
    return (float(np.nanmin(xs) - pad_x), float(np.nanmax(xs) + pad_x)), (
        float(np.nanmin(ys) - pad_y), float(np.nanmax(ys) + pad_y)
    )


def plot_ghost_overlay_slider(bench_frames, user_frames, ghost_dt=0.15, mode='xy'):
    """
    Single subplot overlay:
      - Benchmark in orange
      - User in blue
    Both on the same axes. Slider from t=0..maxTime. We ghost frames within [t-ghost_dt..t+ghost_dt].
    """
    bench_times = [f[0] for f in bench_frames]
    user_times  = [f[0] for f in user_frames]
    max_t = 0.0
    if bench_times: max_t = max(max_t, bench_times[-1])
    if user_times:  max_t = max(max_t, user_times[-1])

    fig, ax = plt.subplots(figsize=(8,6))
    fig.suptitle("Ghost Overlay (Benchmark + User)")

    # Dynamic axis limits from recorded data
    (x_min, x_max), (y_min, y_max) = _compute_axis_limits(bench_frames, user_frames, mode=mode)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.invert_yaxis()  # y grows downwards in image coordinates
    ax.set_title("Overlayed Skeletons")

    ax_slider = fig.add_axes([0.15, 0.02, 0.7, 0.03])
    slider_t = Slider(ax_slider, 'Time', 0.0, max_t, valinit=0.0, valstep=0.01)

    def update(val):
        ax.clear()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.invert_yaxis()
        ax.set_title("Overlayed Skeletons")

        t_center = slider_t.val
        # filter
        bench_sub = [(t,arr) for (t,arr) in bench_frames if (t >= t_center-ghost_dt and t <= t_center+ghost_dt)]
        user_sub  = [(t,arr) for (t,arr) in user_frames  if (t >= t_center-ghost_dt and t <= t_center+ghost_dt)]

        # draw
        for (tb, arrb) in bench_sub:
            coords_2d = side_angle_coords(arrb, mode=mode)
            draw_2d_skeleton(ax, coords_2d, alpha=0.3, color='orange')
        for (tu, arru) in user_sub:
            coords_2d = side_angle_coords(arru, mode=mode)
            draw_2d_skeleton(ax, coords_2d, alpha=0.3, color='blue')

        fig.canvas.draw_idle()

    slider_t.on_changed(update)
    update(0.0)
    plt.tight_layout()
    plt.show()

# ====================== MAIN SCRIPT ======================

def main():
    print("Recording benchmark shot...")
    benchmark_data, frames_bench = record_shot(prompt="Benchmark Shot")
    print("Recording user shot...")
    user_data, frames_user = record_shot(prompt="User Shot")

    # (Optional) shift times so both start at 0
    if frames_bench and frames_user:
        b0 = frames_bench[0][0]
        u0 = frames_user[0][0]
        for i,(t,arr) in enumerate(frames_bench):
            frames_bench[i] = (t - b0, arr)
        for i,(t,arr) in enumerate(frames_user):
            frames_user[i] = (t - u0, arr)

    # Show the single-subplot ghost overlay
    plot_ghost_overlay_slider(frames_bench, frames_user, ghost_dt=0.15, mode='xy')

    # ================= DTW-BASED FORM COMPARISON PLOT =================
    # Build simple overall-form series from angles
    bench_times, bench_form = extract_form_series(benchmark_data)
    user_times,  user_form  = extract_form_series(user_data)

    if len(bench_form) > 1 and len(user_form) > 1:
        # Run DTW on scalar form series
        dist, path = fastdtw(bench_form.reshape(-1,1), user_form.reshape(-1,1), dist=euclidean)

        # Map user indices to closeness 0..100 (100 means identical to benchmark)
        # alpha chosen so that ~50 degrees avg diff -> 0% closeness.
        alpha = 2.0
        user_map = {}
        for (i, j) in path:
            user_map.setdefault(j, []).append(i)
        user_closeness = np.zeros_like(user_form)
        for j in range(len(user_form)):
            if j in user_map:
                i_list = user_map[j]
                i_mid = i_list[len(i_list)//2]
                diff = abs(float(user_form[j]) - float(bench_form[i_mid]))
                score = max(0.0, min(100.0, 100.0 - alpha * diff))
            else:
                score = 100.0
            user_closeness[j] = score

        # Define key events coarsely along each timeline
        def key_events(times):
            if len(times) == 0:
                return []
            idxs = [0,
                    max(0, len(times)//3),
                    max(0, len(times)//2),
                    max(0, (2*len(times))//3),
                    len(times)-1]
            names = ["Start", "Ball Set", "Elbow Above Shoulder", "Release", "Follow Through"]
            return [(names[k], float(times[idxs[k]])) for k in range(len(idxs))]

        bench_events = key_events(bench_times)
        user_events  = key_events(user_times)

        # Align user to benchmark start so markers line up visually
        if bench_events and user_events:
            start_offset = bench_events[0][1] - user_events[0][1]
            user_times_aligned = user_times + start_offset
            user_events_aligned = [(n, t + start_offset) for (n, t) in user_events]
        else:
            user_times_aligned = user_times
            user_events_aligned = user_events

        # Generate feedback first
        feedback_lines = generate_feedback(benchmark_data, user_data, bench_times, user_times_aligned, user_closeness)
        
        # Professional color scheme
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                plt.style.use('default')
        fig = plt.figure(figsize=(20, 7), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2], hspace=0.05, wspace=0.2)
        ax = fig.add_subplot(gs[0])
        ax_feedback = fig.add_subplot(gs[1])
        
        # Professional color palette
        benchmark_color = '#2C3E50'  # Dark blue-gray
        user_color = '#3498DB'       # Professional blue
        grid_color = '#E8E8E8'       # Light gray
        text_color = '#34495E'       # Dark gray
        
        # Main DTW plot with professional styling
        ax.plot(bench_times, np.full_like(bench_times, 100.0), '--', 
                color=benchmark_color, linewidth=2.5, label='Benchmark (100%)', alpha=0.8, zorder=1)
        ax.plot(user_times_aligned, user_closeness, '-', 
                color=user_color, linewidth=2.5, label='Your Shot', alpha=0.9, zorder=2)

        # Professional event markers
        for (n, t) in bench_events:
            ax.scatter([t], [100.0], marker='x', color=benchmark_color, s=120, 
                      linewidths=2.5, zorder=4, alpha=0.9)
            ax.text(t, 105, n, fontsize=10, ha='center', va='bottom', 
                   color=text_color, fontweight='600', family='sans-serif')
        
        # Connect user events with vertical lines
        for (n, t) in user_events_aligned:
            j = int(np.argmin(np.abs(user_times_aligned - t)))
            closeness_val = user_closeness[j]
            ax.scatter([t], [closeness_val], marker='o', color=user_color, s=100, 
                      zorder=4, edgecolors='white', linewidths=2, alpha=0.9)
            # Draw connecting line
            ax.plot([t, t], [closeness_val, 100.0], '--', color=grid_color, 
                   linewidth=1, alpha=0.5, zorder=1)
            # Label only key events
            if n in ("Release", "Follow Through"):
                ax.text(t, closeness_val - 5, n, fontsize=9, ha='center', va='top', 
                       color=user_color, fontweight='600', family='sans-serif')

        # Professional axis styling
        ax.set_title("Shot Form Analysis - DTW Alignment", fontsize=16, fontweight='bold', 
                    color=text_color, pad=20, family='sans-serif')
        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight='500', color=text_color, family='sans-serif')
        ax.set_ylabel("Closeness to Benchmark (%)", fontsize=12, fontweight='500', color=text_color, family='sans-serif')
        ax.set_ylim(0, 110)
        ax.set_facecolor('#FAFAFA')
        ax.grid(True, color=grid_color, linewidth=1, alpha=0.6, linestyle='-', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        
        # Professional legend
        legend = ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True,
                          fontsize=11, framealpha=0.95, edgecolor='none')
        legend.get_frame().set_facecolor('white')
        
        # Format feedback with professional styling
        ax_feedback.axis('off')
        ax_feedback.set_facecolor('#FAFAFA')
        
        # Smart text wrapping function - adjusted for wider panel
        def wrap_text_properly(text, max_chars=55):
            """Wrap text properly, preserving word boundaries"""
            words = text.split()
            if not words:
                return ''
            
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_len = len(word)
                # If adding this word would exceed limit, start new line
                if current_length + word_len + (1 if current_line else 0) > max_chars:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = []
                        current_length = 0
                current_line.append(word)
                current_length += word_len + (1 if len(current_line) > 1 else 0)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            return '\n'.join(lines)
        
        # Format and organize feedback with proper wrapping
        formatted_feedback = []
        for line in feedback_lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle headers
            if line.startswith('==='):
                formatted_feedback.append('')  # Extra space before header
                formatted_feedback.append(line)
                formatted_feedback.append('')  # Extra space after header
            else:
                # Always wrap to ensure proper display
                wrapped = wrap_text_properly(line, max_chars=55)
                formatted_feedback.append(wrapped)
                formatted_feedback.append('')  # Space between items
        
        feedback_text = '\n'.join(formatted_feedback)
        
        # Professional feedback panel header
        ax_feedback.text(0.02, 0.98, 'ANALYSIS & FEEDBACK', transform=ax_feedback.transAxes,
                        fontsize=13, fontweight='bold', color=text_color, 
                        verticalalignment='top', family='sans-serif')
        
        # Add divider line
        ax_feedback.plot([0.02, 0.98], [0.95, 0.95], color=grid_color, 
                        linewidth=1.5, transform=ax_feedback.transAxes)
        
        # Feedback content with proper wrapping and styling
        # Position text box to ensure full visibility - text is pre-wrapped
        ax_feedback.text(0.03, 0.88, feedback_text, transform=ax_feedback.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=12', facecolor='white', 
                                edgecolor=grid_color, linewidth=1.2, alpha=0.95),
                        color=text_color, family='sans-serif')
        
        plt.tight_layout(pad=2.0)
        plt.show()
        
        # Also print to terminal
        print("\n" + "="*60)
        print("FEEDBACK & ANALYSIS")
        print("="*60)
        for line in feedback_lines:
            print(line)
        print("="*60 + "\n")

if __name__ == "__main__":
    main()

