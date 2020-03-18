#open and separate out right or left limbs from multiindex dataframes

def extract_limbs(df,network, limb):
    return df[network][limb]

def likelihood_filter(df,threshold):
    df.loc[df['likelihood']<=threshold] = np.nan
    df2 = df
    df2 = df2.ffill()
    df2 = df2.reset_index(drop=True)
    return df2

def visible_limb_x_velocity_peaks(df,height,distance,direction):
    if direction.upper() == "R":
        forward_x = find_peaks(np.gradient(df['x']),height=height,distance=distance)
        backward_x = find_peaks(-1*np.gradient(df['x']),height=height,distance=distance)
        return forward_x[2], backward_x[2]
    elif direction.upper()=="L":
        forward_x = find_peaks(-1*np.gradient(df['x']),height=height,distance=distance)
        backward_x = find_peaks(np.gradient(df['x']),height=height,distance=distance)
        return forward_x[1], backward_x[1]

def visible_limb_y_velocity_peaks(df,height,distance):
    up_y = find_peaks(-1*np.gradient(df['y']),height=height,distance=distance)
    down_y = find_peaks(np.gradient(df['y']),height=height,distance=distance)
    return up_y[0],down_y[0]

def peak_list_union(list1,list2):
    lst = list(set(list1) | set(list2))
    return lst

def zero_velocity_index(df,coord):
    #give the dataframe where the coord velocity is close to 0
    a = np.gradient(df[coord])
    return df.loc[np.where(np.logical_and(a<=0.9,a>=-0.9))]

def zero_velocity_y_position(direction):
    #index of dataframe where x velocity is close to 0
    v_x_zero_wrist = zero_velocity_index(df_wrist,'x').index
    #index of dataframe where y velocity is close to 0
    v_y_zero_wrist = zero_velocity_index(df_wrist,'y').index

    v_x_zero_fingers = zero_velocity_index(df_fingers,'x').index
    v_y_zero_fingers = zero_velocity_index(df_fingers,'y').index

    v_x_zero_ankle = zero_velocity_index(df_ankle,'x').index
    v_y_zero_ankle = zero_velocity_index(df_ankle,'y').index

    v_x_zero_toes = zero_velocity_index(df_toes,'x').index
    v_y_zero_toes = zero_velocity_index(df_toes,'y').index

    v_zero_front = list((set(v_x_zero_wrist) | set(v_y_zero_wrist))&(set(v_x_zero_fingers)|set(v_y_zero_fingers)))
    v_zero_back = list((set(v_x_zero_ankle)| set(v_y_zero_ankle))&(set(v_x_zero_toes)|set(v_y_zero_toes)))

    y_pos_front = df_fingers["y"][v_zero_front].max()+10
    y_pos_back = df_toes["y"][v_zero_back].max()+10

    return y_pos_front,y_pos_back

def find_y_position_peaks(direction):
    front_peaks = find_peaks(df_fingers['y'],height = y_pos_threshold_front,distance=ydist)
    back_peaks = find_peaks(df_toes['y'],height = y_pos_threshold_back,distance=ydist)
    return front_peaks[0],back_peaks[0]

def all_peaks(df,height,distance,direction)
