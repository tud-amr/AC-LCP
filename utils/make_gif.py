def make_gif(frame_folder, outfolder, videosource):
    import glob
    from PIL import Image
    imagelist = glob.glob(f"{frame_folder}/*.png")
    sortedimagelist = sorted(imagelist, key=lambda x: int(x.split('/')[-1][:-4]))
    sortedimagelist = sortedimagelist[0:len(sortedimagelist):5]
    frames = [Image.open(image) for image in sortedimagelist]
    frame_one = frames[0]
    frame_one.save(outfolder + videosource + ".gif", format="GIF", append_images=frames,
                   save_all=True, duration=125, loop=0)  # duration=250, loop=0)


videosource = ['beliefs','state']
<<<<<<< HEAD
video_foldername = "/home/amr/projects/gym_target_ig/ray_results/policy_versions/attentionVsDeepSets/cte_vel/attention/setTransformer_opt_v2/50x50_env/seed100/2ndphase/videos_exp_1_ckpoint_8030/videos/40_targets_episode_2/"
=======
video_foldername = "/home/alvaroserra/projects/gym_target_ig/ray_results/2ndphase/videos_exp_1_ckpoint_8030/videos/results_with_dynamics/"
>>>>>>> 48d06f3950404f07b4c1c67b0091d7fec4786480
belfolder = video_foldername+'beliefs/'
statefolder = video_foldername + 'state/'

for i, video_folder in enumerate([belfolder, statefolder]):
    make_gif(video_folder, video_foldername, videosource[i])