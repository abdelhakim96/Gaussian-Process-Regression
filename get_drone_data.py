import numpy as np
import matplotlib.pyplot as plt



def calculate_force_estimates_and_obtain_data():
    file_path = "drone_data.txt"

    # Read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the data from the file
    data = []
    for line in lines:
        # Split each line into columns using commas as the delimiter
        columns = line.strip().split(",")

        try:
            # Convert the columns to floats
            columns = list(map(float, columns))
            data.append(columns)
        except ValueError:
            # Skip the line if conversion fails
            continue

    # params
    g = 9.81
    m = 3.0

    # Extract the columns for plotting
    time1 = [row[0] for row in data]
    pos_x = [row[4] for row in data]
    pos_y = [row[5] for row in data]
    pos_z = [row[6] for row in data]

    vel_x = [row[7] for row in data]
    vel_y = [row[8] for row in data]
    vel_z = [row[9] for row in data]

    roll = [row[20] for row in data]
    pitch = [row[21] for row in data]
    yaw = [row[22] for row in data]
    thrust = [row[23] for row in data]

    exec_time_1 = [row[24] for row in data]

    gp_pr_x = [row[29] for row in data]
    gp_pr_y = [row[30] for row in data]
    gp_pr_z = [row[31] for row in data]

    p = [row[14] for row in data]
    q = [row[15] for row in data]
    r = [row[16] for row in data]

    acc_x = [row[17] for row in data]
    acc_y = [row[18] for row in data]
    acc_z = [row[19] for row in data]

    wind_x = [row[26] for row in data]
    wind_y = [row[27] for row in data]
    wind_z = [row[28] for row in data]

    # Convert the extracted columns to floats
    time1 = list(map(float, time1))
    gp_pr_x1 = list(map(float, gp_pr_x))
    pos_x = list(map(float, pos_x))
    wind_x = list(map(float, wind_x))

    acc_x = np.array(acc_x)
    vel_y = np.array(vel_y)
    r = np.array(r)
    q = np.array(q)
    vel_z = np.array(vel_z)
    roll = np.array(roll) * (np.pi / 180)

    pitch = np.array(pitch) * (np.pi / 180)

    # ground truth computation
    gt_x = acc_x - vel_y * r + q * vel_z - g * np.sin(pitch)
    gt_y = acc_y - vel_z * p + r * vel_x - g * np.sin(roll) * np.cos(pitch)



    return [gt_x,gt_y,gp_pr_x,gp_pr_y,pos_x,vel_x]



