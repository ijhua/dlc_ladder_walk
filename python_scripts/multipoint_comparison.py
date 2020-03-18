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
        #df_elbow = extract_limbs(df,"left elbow")
        #back left
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"left toes")

        #filter by likelihood
        #frontleft
        df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
        df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
        #df_elbow = likelihood_filter(df_elbow,likelihood_threshold)
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
        total_steps_fl = len(fl_forward)#-len(fl_backward)
        total_steps_bl = len(bl_forward)#-len(bl_backward)
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
        #df_elbow = extract_limbs(df,"right elbow")
        #back right
        df_ankle = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right ankle")
        df_toes = extract_limbs(df,'DLC_resnet101_LadderWalkFeb13shuffle1_1030000',"right toes")

        #filter by likelihood
        #frontright
        df_wrist = likelihood_filter(df_wrist,likelihood_threshold)
        df_fingers = likelihood_filter(df_fingers,likelihood_threshold)
        #df_elbow = likelihood_filter(df_elbow,likelihood_threshold)
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
        total_steps_fr = len(fr_forward)#-len(fr_backward)
        total_steps_br = len(br_forward)#-len(br_backward)

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

    score_front_l = [subject,date,run,crossing,limb_front,hit_count_fl,slip_count_fl,total_steps_fl]
    score_back_l = [subject,date,run,crossing,limb_back,hit_count_bl,slip_count_bl,total_steps_bl]
    score_front_r = [subject,date,run,crossing,limb_front,hit_count_fr,slip_count_fr,total_steps_fr]
    score_back_r = [subject,date,run,crossing,limb_back,hit_count_br,slip_count_br,total_steps_br]

    scores.append(score_front_l)
    scores.append(score_back_l)
    scores.append(score_front_r)
    scores.append(score_back_r)

score_df = pd.DataFrame(scores,columns=score_cols)
score_df["date"] = pd.to_datetime(score_df["date"])


#human_scores
test_human = pd.read_csv("/home/ml/Documents/LW_Scores_IH.csv")
test_human = test_human.ffill(axis=0)
test_human['date'] = pd.to_datetime(test_human['date'])

all_score = score_df.merge(test_human,on=["subject","date","run","limb"])

all_score.to_csv("/home/ml/Documents/comparison_scores_4_mc_rats.csv")

all_score = all_score.dropna()

diff_df=pd.DataFrame()

diff_df["hit_diff"] = all_score["comp_hits"] - all_score["human_hit"]
diff_df["miss_diff"] = all_score["comp_misses"] - all_score["human_miss"]
diff_df["step_diff"] = all_score["comp_steps"] - all_score["human_steps"]

plt.close()
plt.hist(diff_df["hit_diff"],label='Multipoint Difference',alpha=0.5)
plt.legend()
plt.title("Difference in number of hits")
plt.xlabel("Computation - Human")
plt.ylabel("Number of Runs")
#plt.savefig("/home/ml/Documents/comp_z_human_hit_diff.png")
plt.show()

plt.close()
plt.hist(diff_df["miss_diff"],label='Multipoint Difference',alpha=0.5)
plt.legend()
plt.title("Difference in number of misses")
plt.xlabel("Computation - Human")
plt.ylabel("Number of Runs")
#plt.savefig("/home/ml/Documents/comp_z_human_miss_diff.png")
plt.show()

plt.close()
plt.hist(diff_df["step_diff"],label='Multipoint Difference',alpha=0.5)
plt.legend()
plt.title("Difference in number of steps")
plt.xlabel("Computation - Human")
plt.ylabel("Number of Runs")
#plt.savefig("/home/ml/Documents/comp_z_human_step_diff.png")
plt.show()

calcs=[]

for index,row in all_score.iterrows():
    subject = row['subject']
    if subject == "MC30":
        date1 = dt.datetime(2019,11,12)
    elif subject == "MC70":
        date1 = dt.datetime(2019,3,19)
    elif subject == "MC45":
        date1 = dt.datetime(2019,7,23)
    elif subject == "MC61":
        date1 = dt.datetime(2019,6,11)
    elif subject == "MC78":
            date1 = dt.datetime(2019,4,2)
    elif subject == "MC87":
        date1 = dt.datetime(2018,12,17)
    week_num = (row['date'] - date1).days/7
    if week_num <=0:
        week = "Preinjury"
    if week_num>0:
        week="Postinjury"
    limb = row['limb']
    comp_score = row["comp_misses"]/row["comp_steps"]*100
    comp_steps = row["comp_steps"]
    comp_slips = row["comp_misses"]
    human_score = row["human_miss"]/row["human_steps"]*100
    human_steps = row["human_steps"]
    human_miss = row["human_miss"]
    calcs.append([subject,week,limb,comp_score,comp_steps,comp_slips,human_score,human_steps,human_miss])
calc_df = pd.DataFrame(calcs,columns=["subject","week","limb","comp_score","comp_steps","comp_misses","human_score","human_steps","human_misses"])
#calc_df = calc_df.round({"week":0})
df_new = calc_df.groupby(["week","limb"])["comp_score","comp_steps","comp_misses","human_score","human_steps","human_misses"].agg(["mean",'sem'])

df_new=df_new.reset_index()
df_new = df_new.sort_values(by=["week"])

fd = df_new.loc[df_new["limb"] == "Dominant Front"]

fn = df_new.loc[df_new["limb"] =="Nondominant Front"]

bd = df_new.loc[df_new["limb"] =="Dominant Back"]

bn = df_new.loc[df_new["limb"] =="Nondominant Back"]

limbs = [fd,fn,bd,bn]
for limb in limbs:
    limb = limb.reset_index()
    name = limb["limb"][0]
    plt.close()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_score"]["mean"],yerr=limb["comp_score"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_score"]["mean"],yerr=limb["human_score"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" MC Rats")
    plt.xlabel("Week")
    plt.ylabel("%slip")
    plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    #plt.savefig("/home/ml/Documents/"+name+"_weekly_number_score_MC_rats.png")
    plt.show()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_steps"]["mean"],yerr=limb["comp_steps"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_steps"]["mean"],yerr=limb["human_steps"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" MC Rats")
    plt.xlabel("Week")
    plt.ylabel("Number of Steps")
    plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    #plt.savefig("/home/ml/Documents/"+name+"_weekly_step_count_MC_rats.png")
    plt.show()
    plt.figure()
    plt.rc('xtick')
    plt.rc('ytick')
    plt.errorbar(limb["week"],limb["comp_misses"]["mean"],yerr=limb["comp_misses"]["sem"] , uplims=True, lolims=True,label="Computational")
    plt.errorbar(limb["week"],limb["human_misses"]["mean"],yerr=limb["human_misses"]["sem"] , uplims=True, lolims=True,label="Human")
    plt.title( name+" MC Rats")
    plt.xlabel("Week")
    plt.ylabel("Number of Slips")
    plt.ylim(bottom=0)
    plt.legend()
    plt.gca().invert_xaxis()
    #plt.savefig("/home/ml/Documents/"+name+"_weekly_slip_count_MC_rats.png")
    plt.show()
