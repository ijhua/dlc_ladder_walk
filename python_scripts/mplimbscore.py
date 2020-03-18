# read the file
rats = ["MC45","MC61","MC78","MC87","MC30","MC70"]
right_handed = ["MC45","MC61","MC78","MC87","MC30","MC70"]
left_handed = []
folders = []
for rat in rats:
    folders+=glob.glob("/home/ml/Documents/Not_TADSS_Videos/"+rat+"/cut/dlc_output_16-810/*.h5")

scores = []
score_cols = ["subject", "date", "run", "crossing","limb","comp_hits","comp_misses","comp_steps"]
for f in folders:
    df = pd.read_hdf(f)
    name=f.split("/")[8]
    run = name.split("_")[2]
    subject = name.split("_")[0]
    date = name.split("_")[1]
    crossing=name.split("_")[3][-1]
    likelihood_threshold = 0.1
    xheight = 25
    xdist = 5
    yheight = 5
    ydist = 4

    #left crossings
    if run[0] == "L":
        if subject in right_handed:
            limb_front = "Nondominant Front"
            limb_back = "Nondominant Back"
        elif subject in left_handed:
            limb_front = "Dominant Front"
            limb_back = "Dominant Back"
        #split all the limbs
        #frontleft
        df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left fingers")
        df_elbow = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left elbow")
        #back left
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left toes")

        #filter by likelihood
        #frontleft
        df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
        df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
        df_elbow = likelihood_filter(df_elbow,likelihood_threshold)
        #back left
        df_ankle = likelihood_filter(df_ankle,likelihood_threshold)
        df_toes = likelihood_filter(df_toes,likelihood_threshold)

        #get the x velocity peaks
        wrist_forward_list,wrist_backward_list = visible_limb_x_velocity_peaks(df_wrist,xheight,xdist,"L")
        fingers_forward_list,fingers_backward_list = visible_limb_x_velocity_peaks(df_fingers,xheight,xdist,"L")

        ankle_forward_list,ankle_backward_list = visible_limb_x_velocity_peaks(df_ankle,xheight,xdist,"L")
        toes_forward_list,toes_backward_list = visible_limb_x_velocity_peaks(df_toes,xheight,xdist,"L")

        #join the two points on the front left limb
        fl_forward = peak_list_union(wrist_forward_list,fingers_forward_list)
        fl_backward = peak_list_union(wrist_backward_list,fingers_backward_list)

        #join two points on the back left limb
        bl_forward = peak_list_union(ankle_forward_list,toes_forward_list)
        bl_backward = peak_list_union(ankle_backward_list,toes_backward_list)

        #get y velocity peaks
        wrist_up_list, wrist_down_list = visible_limb_y_velocity_peaks(df_wrist,yheight,ydist)
        fingers_up_list, fingers_down_list = visible_limb_y_velocity_peaks(df_fingers,yheight,ydist)

        ankle_up_list, ankle_down_list = visible_limb_y_velocity_peaks(df_ankle,yheight,ydist)
        toes_up_list, toes_down_list = visible_limb_y_velocity_peaks(df_toes,yheight,ydist)

        #join front lists
        fl_up = peak_list_union(wrist_up_list,fingers_up_list)
        fl_down = peak_list_union(wrist_down_list,fingers_down_list)
        #join back lists
        bl_up = peak_list_union(ankle_up_list,toes_up_list)
        bl_down = peak_list_union(ankle_down_list,toes_down_list)

        #number of x peaks is the number of total steps. I don't think there are really ever any backward peaks, so we'll just ignore them for now
        total_steps_fl = len(fl_forward)
        total_steps_bl = len(bl_forward)
        total_steps_fr = np.nan
        total_steps_br = np.nan


        #figure out the y position threshold. it'll be when vx and vy are approximately 0
        y_pos_threshold_front,y_pos_threshold_back = zero_velocity_y_position("L")

        y_pos_peaks_front,y_pos_peaks_back = find_y_position_peaks("L")
        slip_count_fl = len(y_pos_peaks_front)
        slip_count_bl = len(y_pos_peaks_back)
        slip_count_fr = np.nan
        slip_count_br = np.nan

        hit_count_fl = total_steps_fl - slip_count_fl
        hit_count_bl = total_steps_bl - slip_count_bl
        hit_count_fr = np.nan
        hit_count_br = np.nan

        plt.close()
        plt.plot(df_fingers['x'],df_fingers['y'],label="Visible Front Limb")
        plt.plot([df_fingers["x"].min(),df_fingers['x'].max()],[y_pos_threshold_front,y_pos_threshold_front],label="Rung")
        plt.scatter(df_fingers['x'][fl_forward],df_fingers['y'][fl_forward],color='b',label='Vx peaks')
        plt.scatter(df_fingers['x'][y_pos_peaks_front],df_fingers['y'][y_pos_peaks_front],color='r',label='Peaks below rungs')
        plt.legend()
        plt.ylabel("y")
        plt.xlabel("x"+" Steps: "+str(total_steps_fl)+" Slips: "+str(slip_count_fl))
        plt.gca().invert_yaxis()
        plt.savefig("/home/ml/Documents/multipoint_stepfinding_figures_front/"+subject+"_"+date+"_"+run+"_front.png")
        plt.close()
        plt.plot(df_toes['x'],df_toes['y'],label="Visible Back Limb")
        plt.plot([df_toes["x"].min(),df_toes['x'].max()],[y_pos_threshold_back,y_pos_threshold_back],label="Rung")
        plt.scatter(df_toes['x'][bl_forward],df_toes['y'][bl_forward],color='b',label='Vx peaks')
        plt.scatter(df_toes['x'][y_pos_peaks_back],df_toes['y'][y_pos_peaks_back],color='r',label='Peaks below rungs')
        plt.legend()
        plt.ylabel("y")
        plt.xlabel("x"+" Steps: "+str(total_steps_bl)+" Slips: "+str(slip_count_bl))
        plt.gca().invert_yaxis()
        plt.savefig("/home/ml/Documents/multipoint_stepfinding_figures_back/"+subject+"_"+date+"_"+run+"_back.png")
        plt.close()
    #right side
    if run[0] == "R":
        if subject in left_handed:
            limb_front = "Nondominant Front"
            limb_back = "Nondominant Back"
        elif subject in right_handed:
            limb_front = "Dominant Front"
            limb_back = "Dominant Back"
        #split all the limbs
        #frontright
        df_wrist = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right wrist")
        df_fingers = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right fingers")
        df_elbow = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right elbow")
        #back right
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right toes")

        #filter by likelihood
        #frontright
        df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
        df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
        df_elbow = likelihood_filter(df_elbow,likelihood_threshold)
        #back right
        df_ankle = likelihood_filter(df_ankle,likelihood_threshold)
        df_toes = likelihood_filter(df_toes,likelihood_threshold)

        #get the x velocity peaks
        wrist_forward_list,wrist_backward_list = visible_limb_x_velocity_peaks(df_wrist,xheight,xdist,"R")
        fingers_forward_list,fingers_backward_list = visible_limb_x_velocity_peaks(df_fingers,xheight,xdist,"R")

        ankle_forward_list,ankle_backward_list = visible_limb_x_velocity_peaks(df_ankle,xheight,xdist,"R")
        toes_forward_list,toes_backward_list = visible_limb_x_velocity_peaks(df_toes,xheight,xdist,"R")

        #join the two points on the front right limb
        fr_forward = peak_list_union(wrist_forward_list,fingers_forward_list)
        fr_backward = peak_list_union(wrist_backward_list,fingers_backward_list)

        #join two points on the back right limb
        br_forward = peak_list_union(ankle_forward_list,toes_forward_list)
        br_backward = peak_list_union(ankle_backward_list,toes_backward_list)

        #get y velocity peaks
        wrist_up_list, wrist_down_list = visible_limb_y_velocity_peaks(df_wrist,yheight,ydist)
        fingers_up_list, fingers_down_list = visible_limb_y_velocity_peaks(df_fingers,yheight,ydist)

        ankle_up_list, ankle_down_list = visible_limb_y_velocity_peaks(df_ankle,yheight,ydist)
        toes_up_list, toes_down_list = visible_limb_y_velocity_peaks(df_toes,yheight,ydist)

        #join front lists
        fr_up = peak_list_union(wrist_up_list,fingers_up_list)
        fr_down = peak_list_union(wrist_down_list,fingers_down_list)
        #join back lists
        br_up = peak_list_union(ankle_up_list,toes_up_list)
        br_down = peak_list_union(ankle_down_list,toes_down_list)

        #number of x peaks is the number of total steps. I don't think there are really ever any backward peaks, so we'll just ignore them for now
        total_steps_fl = np.nan
        total_steps_bl = np.nan
        total_steps_fr = len(fr_forward)
        total_steps_br = len(br_forward)

        #figure out the y position threshold. it'll be when vx and vy are approximately 0
        y_pos_threshold_front,y_pos_threshold_back = zero_velocity_y_position("r")

        y_pos_peaks_front,y_pos_peaks_back = find_y_position_peaks("r")
        slip_count_fl = np.nan
        slip_count_bl = np.nan
        slip_count_fr = len(y_pos_peaks_front)
        slip_count_br = len(y_pos_peaks_back)

        hit_count_fl = np.nan
        hit_count_bl = np.nan
        hit_count_fr = total_steps_fr - slip_count_fr
        hit_count_br = total_steps_br - slip_count_br


        plt.close()
        plt.plot(df_fingers['x'],df_fingers['y'],label="Visible Front Limb")
        plt.plot([df_fingers["x"].min(),df_fingers['x'].max()],[y_pos_threshold_front,y_pos_threshold_front],label="Rung")
        plt.scatter(df_fingers['x'][fr_forward],df_fingers['y'][fr_forward],color='b',label='Vx peaks')
        plt.scatter(df_fingers['x'][y_pos_peaks_front],df_fingers['y'][y_pos_peaks_front],color='r',label='Peaks below rungs')
        plt.legend()
        plt.ylabel("y")
        plt.xlabel("x"+" Steps: "+str(total_steps_fr)+" Slips: "+str(slip_count_fr))
        plt.gca().invert_yaxis()
        plt.savefig("/home/ml/Documents/multipoint_stepfinding_figures_front/"+subject+"_"+date+"_"+run+"_front.png")
        plt.close()
        plt.plot(df_toes['x'],df_toes['y'],label="Visible Back Limb")
        plt.plot([df_toes["x"].min(),df_toes['x'].max()],[y_pos_threshold_back,y_pos_threshold_back],label="Rung")
        plt.scatter(df_toes['x'][br_forward],df_toes['y'][br_forward],color='b',label='Vx peaks')
        plt.scatter(df_toes['x'][y_pos_peaks_back],df_toes['y'][y_pos_peaks_back],color='r',label='Peaks below rungs')
        plt.legend()
        plt.ylabel("y")
        plt.xlabel("x"+" Steps: "+str(total_steps_br)+" Slips: "+str(slip_count_br))
        plt.gca().invert_yaxis()
        plt.savefig("/home/ml/Documents/multipoint_stepfinding_figures_back/"+subject+"_"+date+"_"+run+"_back.png")
        plt.close()
    score_front_l = [subject,date,run,crossing,limb_front,hit_count_fl,slip_count_fl,total_steps_fl]
    score_back_l = [subject,date,run,crossing,limb_back,hit_count_bl,slip_count_bl,total_steps_bl]
    score_front_r = [subject,date,run,crossing,limb_front,hit_count_fr,slip_count_fr,total_steps_fr]
    score_back_r = [subject,date,run,crossing,limb_back,hit_count_br,slip_count_br,total_steps_br]

    scores.append(score_front_l)
    scores.append(score_back_l)
    scores.append(score_front_r)
    scores.append(score_back_r)


    '''video = f.split(".")[0]+"_labeled.mp4"
    cap = cv2.VideoCapture(video)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    front_output = "/home/ml/Documents/multipoint_stepfinding_videos_front/"+subject+"_"+date+"_"+run+"_front.mp4"
    subprocess.call(["ffmpeg", "-y","-i", video, "-vf", "drawbox=x=0:y="+str(y_pos_threshold_front)+":w=1920:h=5:color=red",front_output])
    back_output = "/home/ml/Documents/multipoint_stepfinding_videos_back/"+subject+"_"+date+"_"+run+"_back.mp4"
    subprocess.call(["ffmpeg", "-y","-i", video, "-vf", "drawbox=x=0:y="+str(y_pos_threshold_back)+":w=1920:h=5:color=red",back_output])'''
    cap = cv2.VideoCapture(f.split(".")[0]+"_labeled.mp4")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = f.split(".")[0]+"_labeled.mp4"
    vel_video = "/home/ml/Documents/MC_rats/graphs/"+name.split("_")[0]+"_"+name.split("_")[1]+"_"+name.split("_")[2]+"_graph.mp4"
    out_file = "/home/ml/Documents/MC_rats/"+subject+"_"+date+"_"+run+"_annotated.mp4"
    out_file2 = out_file.split(".")[0]+"_rat_slow.mp4"

    if os.path.exists(out_file):
        continue
    else:
        for i in range(frames):
           plt.close()
           plt.figure(figsize=(20,10))
           fig,(ax1,ax2) = plt.subplots(2)
           ax1.plot(np.gradient(df_fingers['x'])[list(range(i+1))],label="front")
           ax1.plot(np.gradient(df_toes['x'])[list(range(i+1))],label="back")
           if run[0] == "R":
               ax1.scatter(df_fingers.index[fr_forward],np.gradient(df_fingers['x'])[fr_forward],color='b')
               ax1.scatter(df_toes.index[br_forward],np.gradient(df_toes['x'])[br_forward],color='cyan')
           elif run[0] == "L":
               ax1.scatter(df_fingers.index[fl_forward],np.gradient(df_fingers['x'])[fl_forward],color='b')
               ax1.scatter(df_toes.index[bl_forward],np.gradient(df_toes['x'])[bl_forward],color='cyan')
           ax1.set_xlabel("x frame")
           ax1.set_ylabel("x velocity")
           ax1.set_title("Step Counting")
           ax2.plot(np.gradient(df_fingers['y'])[list(range(i+1))],label="front")
           ax2.plot(np.gradient(df_toes['y'])[list(range(i+1))],label="back")
           ax2.scatter(df_fingers.index[y_pos_peaks_front],np.gradient(df_fingers['y'])[y_pos_peaks_front],color='b')
           ax2.scatter(df_toes.index[y_pos_peaks_back],np.gradient(df_toes['y'])[y_pos_peaks_back],color='cyan')
           ax2.set_xlabel("y frame")
           ax2.set_ylabel("y velocity")
           ax2.set_title("Slip Counting")
           plt.legend()
           ax1.set_xlim(0,df_fingers.index.max())
           ax2.set_xlim(0,df_fingers.index.max())
           fig.tight_layout()
           plt.savefig("/home/ml/Documents/MC_rats/temp/"+str(i).zfill(3)+".png")
           plt.close()

        subprocess.call(["ffmpeg","-pattern_type","glob","-i","/home/ml/Documents/MC_rats/temp/*.png","-r","24","-pix_fmt","yuv420p","-s","1920x1080",vel_video])

        subprocess.call("rm /home/ml/Documents/MC_rats/temp/*.png",shell=True)
        subprocess.call(["ffmpeg","-y", "-i", video , "-i", vel_video, "-filter_complex","vstack=inputs=2", out_file])
        #drawbox=x=0:y="+str(y_pos_threshold_front)+":w=1920:h=4:color=red,drawbox=x=0:y="+str(y_pos_threshold_back)+":w=1920:h=4:color=green,
        subprocess.call(['ffmpeg',"-y",'-i',out_file,"-filter:v","setpts=8*PTS",out_file2])


score_df = pd.DataFrame(scores,columns=score_cols)
score_df["date"] = pd.to_datetime(score_df["date"])
score_df.to_csv("/home/ml/Documents/multipoint_scores_all_rats_intersection_2.csv")
