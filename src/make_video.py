from moviepy.editor import *
import glob
import subprocess


sequence = 'boreas-2021-11-02-11-16'
result_path = '/media/data/RadarOSM/src/results/visualization/' + sequence 
print(result_path)




if 'boreas' in sequence:
	cart_path = '/media/data/RadarOSM/data/radar/Boreas_Radar_Dataset/' + sequence + '_cart/'
elif 'mulran' in sequence:
	cart_path = '/media/data/RadarOSM/data/radar/MulRan_Dataset/' + sequence + '_cart/'
elif 'oxford' in sequence:
	cart_path = '/media/data/RadarOSM/data/radar/Oxford_Dataset/' + sequence + '_cart/'
else:
	print ('New datasets not included!!!')




post_img_files = sorted(glob.glob(result_path + '/*posterior.png'))
first_img = post_img_files[0]
first_idx = first_img.index('posterior.png')
frame_idx = int(first_img[first_idx-7:first_idx-1])
print(frame_idx)
print(first_idx)

# Cartesian images
command = ('ffmpeg', '-r', '50', '-y', '-start_number', str(frame_idx),
           '-i', cart_path + '%06d.jpg',
                   '-vcodec', 'mpeg4', '-vtag', 'xvid', '-b', '3500k',
                   sequence + '_cart_movie.avi')
subprocess.call(command)

# Posterior
post_clip = ImageSequenceClip(post_img_files, fps = 10) 
post_clip.write_videofile(sequence + "_posterior_movie.mp4", bitrate='3500k')

# Observation
observation_img_files = sorted(glob.glob(result_path + '/*debug_point_line_normal_????.png'))
observation_clip = ImageSequenceClip(observation_img_files, fps = 10) 
observation_clip.write_videofile(sequence + "_obseravtion_movie.mp4", bitrate='3500k')

# Trajectory
traj_img_files = sorted(glob.glob(result_path + '/*trajectories_????.png'))
traj_clip = ImageSequenceClip(traj_img_files, fps = 10) 
traj_clip.write_videofile(sequence + "_trajectory_movie.mp4", bitrate='3500k')



