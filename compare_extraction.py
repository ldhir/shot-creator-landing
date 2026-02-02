#!/usr/bin/env python3
"""
Compare user extraction JSON with Anthony Edwards benchmark data.
Usage: python compare_extraction.py <user_json_file>
   or: python compare_extraction.py (then paste JSON when prompted)
"""

import json
import re
import sys
import math

def non_linear_similarity(diff, max_diff, exponent=2):
    """Calculate non-linear similarity score (0-100)"""
    if diff >= max_diff:
        return 0
    normalized = diff / max_diff
    return max(0, (1 - math.pow(normalized, exponent)) * 100)

def extract_metrics_from_data(data):
    """Extract averaged metrics from shot data (matches JavaScript logic)"""
    metrics = {
        'release_height': None,
        'wrist_snap': None,
        'elbow_extension': None,
        'foot_alignment': None,
        'trunk_lean': None,
        'knee_bend': None,
        'elbow_flare': None,
        'shoulder_angle': None,
        'foot_angle': None
    }
    
    if not data or len(data) == 0:
        return metrics
    
    # Find key frames
    first_pre_shot_frame = None
    first_pre_shot_index = -1
    first_follow_through_frame = None
    first_follow_through_index = -1
    release_point_frame = None
    max_release_height = -float('inf')
    
    # Find first pre_shot frame that is part of a valid shot sequence
    for i, frame in enumerate(data):
        if frame.get('state') == 'pre_shot':
            # Check if this pre_shot is followed by follow_through (valid shot sequence)
            found_follow_through = False
            sequence_broken = False
            follow_through_index = -1
            
            for j in range(i + 1, len(data)):
                if data[j].get('state') == 'follow_through':
                    found_follow_through = True
                    follow_through_index = j
                    break
                elif data[j].get('state') == 'neutral':
                    sequence_broken = True
                    break
            
            if found_follow_through and not sequence_broken:
                first_pre_shot_frame = frame
                first_pre_shot_index = i
                print(f'Found valid pre_shot at frame {i}, followed by follow_through at frame {follow_through_index}')
                break
            elif sequence_broken:
                print(f'Skipping pre_shot at frame {i} - sequence broken (hit neutral before follow_through)')
    
    if not first_pre_shot_frame:
        print('Warning: No valid pre_shot frame found that is followed by follow_through')
    
    # Find first follow_through frame
    start_index = first_pre_shot_index if first_pre_shot_index >= 0 else 0
    for i in range(start_index, len(data)):
        if data[i].get('state') == 'follow_through':
            first_follow_through_frame = data[i]
            first_follow_through_index = i
            break
    
    # Find release point (frame with maximum release_height)
    for frame in data:
        frame_metrics = frame.get('metrics', {})
        release_height = frame_metrics.get('release_height')
        if release_height is not None and not math.isnan(release_height):
            if release_height > max_release_height:
                max_release_height = release_height
                release_point_frame = frame
    
    # If no release point found, use first follow_through frame
    if not release_point_frame and first_follow_through_frame:
        release_point_frame = first_follow_through_frame
    
    # Calculate metrics at specific frames
    # 1. Elbow flare, knee bend, trunk lean: at first pre_shot frame
    if first_pre_shot_frame:
        frame_metrics = first_pre_shot_frame.get('metrics', {})
        metrics['elbow_flare'] = frame_metrics.get('elbow_flare') if frame_metrics.get('elbow_flare') is not None else None
        metrics['knee_bend'] = frame_metrics.get('knee_bend') if frame_metrics.get('knee_bend') is not None else None
        metrics['trunk_lean'] = frame_metrics.get('trunk_lean') if frame_metrics.get('trunk_lean') is not None else None
    
    # 2. Elbow extension: (elbow angle at start of pre_shot) - (elbow angle at start of follow_through)
    if first_pre_shot_frame and first_follow_through_frame:
        pre_shot_metrics = first_pre_shot_frame.get('metrics', {})
        follow_through_metrics = first_follow_through_frame.get('metrics', {})
        pre_shot_elbow_ext = pre_shot_metrics.get('elbow_extension')
        follow_through_elbow_ext = follow_through_metrics.get('elbow_extension')
        
        if (pre_shot_elbow_ext is not None and not math.isnan(pre_shot_elbow_ext) and
            follow_through_elbow_ext is not None and not math.isnan(follow_through_elbow_ext)):
            metrics['elbow_extension'] = pre_shot_elbow_ext - follow_through_elbow_ext
    
    # 3. Wrist snap: use value at release point (or follow_through if release point not available)
    if release_point_frame:
        frame_metrics = release_point_frame.get('metrics', {})
        release_point_wrist_snap = frame_metrics.get('wrist_snap')
        if release_point_wrist_snap is not None and not math.isnan(release_point_wrist_snap):
            metrics['wrist_snap'] = release_point_wrist_snap
    
    # Fallback to follow_through frame if release point not available
    if metrics['wrist_snap'] is None and first_follow_through_frame:
        frame_metrics = first_follow_through_frame.get('metrics', {})
        follow_through_wrist_snap = frame_metrics.get('wrist_snap')
        if follow_through_wrist_snap is not None and not math.isnan(follow_through_wrist_snap):
            metrics['wrist_snap'] = follow_through_wrist_snap
    
    # 4. Release height: maximum value
    if max_release_height != -float('inf'):
        metrics['release_height'] = max_release_height
    
    # 5. Foot alignment, shoulder_angle, foot_angle: at first follow_through frame
    if first_follow_through_frame:
        frame_metrics = first_follow_through_frame.get('metrics', {})
        metrics['foot_alignment'] = frame_metrics.get('foot_alignment') if frame_metrics.get('foot_alignment') is not None else None
        metrics['shoulder_angle'] = frame_metrics.get('shoulder_angle') if frame_metrics.get('shoulder_angle') is not None else None
        metrics['foot_angle'] = frame_metrics.get('foot_angle') if frame_metrics.get('foot_angle') is not None else None
    
    # Fallback to averages for any metrics that couldn't be calculated at specific frames
    sums = {}
    counts = {}
    
    for frame in data:
        frame_metrics = frame.get('metrics', {})
        if frame_metrics:
            for metric_name in metrics:
                if metrics[metric_name] is not None:
                    continue  # Skip if already calculated
                
                value = frame_metrics.get(metric_name)
                if value is not None and not math.isnan(value):
                    if metric_name not in sums:
                        sums[metric_name] = 0
                        counts[metric_name] = 0
                    sums[metric_name] += value
                    counts[metric_name] += 1
    
    # Calculate averages
    for metric_name in metrics:
        if metrics[metric_name] is None and counts.get(metric_name, 0) > 0:
            metrics[metric_name] = sums[metric_name] / counts[metric_name]
    
    return metrics

def compare_detailed_metrics(user_data, benchmark_data):
    """Compare detailed metrics between user and benchmark data"""
    if not user_data or not benchmark_data or len(user_data) == 0 or len(benchmark_data) == 0:
        return {'overall_score': 0, 'metric_scores': {}, 'shared_traits': [], 'differences': []}
    
    # Metric weights
    weights = {
        'release_height': 0.24,
        'wrist_snap': 0.18,
        'elbow_extension': 0.18,
        'trunk_lean': 0.12,
        'knee_bend': 0.12,
        'elbow_flare': 0.12,
        'shoulder_angle': 0.04
    }
    
    # Maximum differences for 0% similarity
    max_diffs = {
        'release_height': 0.3,
        'wrist_snap': 45,
        'elbow_extension': 40,
        'trunk_lean': 25,
        'knee_bend': 40,
        'elbow_flare': 30,
        'shoulder_angle': 45
    }
    
    # Non-linearity exponents
    exponents = {
        'release_height': 1.5,
        'wrist_snap': 2.0,
        'elbow_extension': 2.0,
        'trunk_lean': 2.0,
        'knee_bend': 1.8,
        'elbow_flare': 2.2,
        'shoulder_angle': 1.5
    }
    
    # Extract metrics
    user_metrics = extract_metrics_from_data(user_data)
    benchmark_metrics = extract_metrics_from_data(benchmark_data)
    
    print('\n=== Extracted Metrics ===')
    print('User metrics:', json.dumps(user_metrics, indent=2))
    print('\nBenchmark metrics:', json.dumps(benchmark_metrics, indent=2))
    
    metric_scores = {}
    metric_diffs = {}
    
    # Compare each metric
    for metric_name in weights:
        user_value = user_metrics.get(metric_name)
        benchmark_value = benchmark_metrics.get(metric_name)
        
        if user_value is None or benchmark_value is None:
            metric_scores[metric_name] = None
            metric_diffs[metric_name] = None
            continue
        
        diff = abs(user_value - benchmark_value)
        metric_diffs[metric_name] = diff
        
        max_diff = max_diffs[metric_name]
        exponent = exponents[metric_name]
        score = non_linear_similarity(diff, max_diff, exponent)
        metric_scores[metric_name] = score
        
        if score == 0:
            print(f'⚠️ {metric_name}: 0% similarity - User: {user_value:.2f}, Benchmark: {benchmark_value:.2f}, Diff: {diff:.2f}, MaxDiff: {max_diff}')
    
    # Calculate weighted overall score
    total_weight = 0
    weighted_sum = 0
    
    for metric_name in weights:
        if metric_scores[metric_name] is not None:
            weight = weights[metric_name]
            total_weight += weight
            weighted_sum += metric_scores[metric_name] * weight
    
    overall_score = weighted_sum / total_weight if total_weight > 0 else 0
    
    # Identify shared traits and differences
    shared_traits = []
    differences = []
    
    for metric_name in weights:
        if metric_scores[metric_name] is not None:
            score = metric_scores[metric_name]
            diff = metric_diffs[metric_name]
            
            metric_info = {
                'name': metric_name,
                'score': score,
                'difference': diff,
                'user_value': user_metrics[metric_name],
                'benchmark_value': benchmark_metrics[metric_name]
            }
            
            if score >= 85:
                shared_traits.append(metric_info)
            elif score < 70:
                differences.append(metric_info)
    
    shared_traits.sort(key=lambda x: x['score'], reverse=True)
    differences.sort(key=lambda x: x['score'])
    
    return {
        'overall_score': overall_score,
        'metric_scores': metric_scores,
        'metric_diffs': metric_diffs,
        'shared_traits': shared_traits,
        'differences': differences,
        'user_metrics': user_metrics,
        'benchmark_metrics': benchmark_metrics
    }

def load_benchmark_file(filepath):
    """Load benchmark data from JavaScript file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the data array from the JavaScript file
    # Match any const <name>_data = [...] pattern
    match = re.search(r'const \w+_data = (\[[\s\S]*?\]);', content)
    if not match:
        raise ValueError(f'Could not parse JavaScript file: {filepath}. Expected format: const <name>_data = [...];')
    
    # Use eval to parse the JavaScript array (safe in this context)
    data_str = match.group(1)
    # Replace JavaScript null with Python None
    data_str = data_str.replace('null', 'None')
    data = eval(data_str)
    
    return data

def main():
    # Load benchmark data
    benchmark_path = 'tool/player_data/anthony_edwards.js'
    print(f'Loading benchmark from {benchmark_path}...')
    try:
        benchmark_data = load_benchmark_file(benchmark_path)
        print(f'✓ Loaded {len(benchmark_data)} benchmark frames')
    except Exception as e:
        print(f'Error loading benchmark: {e}')
        return
    
    # Load user data
    if len(sys.argv) > 1:
        # Load from file
        user_file_path = sys.argv[1]
        print(f'\nLoading user extraction from {user_file_path}...')
        try:
            # Check if it's a JavaScript file or JSON file
            if user_file_path.endswith('.js'):
                # Parse JavaScript file (same format as benchmark)
                user_data = load_benchmark_file(user_file_path)
            else:
                # Parse JSON file
                with open(user_file_path, 'r') as f:
                    user_data_obj = json.load(f)
                # Handle both formats: direct array or object with 'frames' key
                user_data = user_data_obj.get('frames', user_data_obj) if isinstance(user_data_obj, dict) else user_data_obj
                if not isinstance(user_data, list):
                    raise ValueError('User data must be an array or object with "frames" array')
            print(f'✓ Loaded {len(user_data)} user frames')
        except Exception as e:
            print(f'Error loading user file: {e}')
            import traceback
            traceback.print_exc()
            return
    else:
        # Read from stdin
        print('\nPaste your JSON extraction data (press Ctrl+D when done):')
        try:
            json_str = sys.stdin.read()
            user_data_obj = json.loads(json_str)
            user_data = user_data_obj.get('frames', user_data_obj) if isinstance(user_data_obj, dict) else user_data_obj
            if not isinstance(user_data, list):
                raise ValueError('User data must be an array or object with "frames" array')
            print(f'✓ Loaded {len(user_data)} user frames')
        except Exception as e:
            print(f'Error parsing JSON: {e}')
            return
    
    # Perform comparison
    print('\n=== Performing Comparison ===')
    comparison = compare_detailed_metrics(user_data, benchmark_data)
    
    # Display results
    print('\n' + '='*60)
    print('COMPARISON RESULTS')
    print('='*60)
    print(f'\nOverall Similarity Score: {comparison["overall_score"]:.1f}%')
    
    print('\n--- Individual Metric Scores ---')
    for metric_name, score in comparison['metric_scores'].items():
        if score is not None:
            diff = comparison['metric_diffs'][metric_name]
            user_val = comparison['user_metrics'][metric_name]
            bench_val = comparison['benchmark_metrics'][metric_name]
            print(f'{metric_name:20s}: {score:5.1f}% (User: {user_val:7.2f}, Benchmark: {bench_val:7.2f}, Diff: {diff:6.2f})')
        else:
            print(f'{metric_name:20s}: N/A (missing data)')
    
    if comparison['shared_traits']:
        print('\n--- Shared Traits (High Similarity ≥85%) ---')
        for trait in comparison['shared_traits']:
            print(f'{trait["name"]:20s}: {trait["score"]:5.1f}% (diff: {trait["difference"]:.2f})')
    
    if comparison['differences']:
        print('\n--- Key Differences (Need Improvement <70%) ---')
        for diff in comparison['differences']:
            print(f'{diff["name"]:20s}: {diff["score"]:5.1f}% (diff: {diff["difference"]:.2f})')
            print(f'  User: {diff["user_value"]:.2f}, Benchmark: {diff["benchmark_value"]:.2f}')
    
    print('\n' + '='*60)

if __name__ == '__main__':
    main()
