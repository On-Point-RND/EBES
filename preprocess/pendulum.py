from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import trange
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StructType, ArrayType, StructField, LongType


# data generation from https://arxiv.org/abs/2401.15935

# Initial and end values
st = 0  # Start time (s)
# et = 5  # End time (s)
et_min = 3  # End time (s)
et_max = 5  # End time (s)

ts = 0.1  # Time step (s)
g = 9.81  # Acceleration due to gravity (m/s^2)
# b = 0.5  # Damping factor (kg/s)
m = 1  # Mass of bob (kg)
mean_n = 30
alpha = 0.5
train_size = 80_000
test_size = 20_000


def hawkes_intensity(mu, alpha, points, t):
    """Find the hawkes intensity:
    mu + alpha * sum( np.exp(-(t-s)) for s in points if s<=t )
    """
    p = np.array(points)
    p = p[p <= t]
    p = np.exp(p - t) * alpha
    return mu + np.sum(p)
    # return mu + alpha * sum( np.exp(s - t) for s in points if s <= t )


def simulate_hawkes(mu, alpha, st, et):
    t = st
    points = []
    all_samples = []
    while t < et:
        m = hawkes_intensity(mu, alpha, points, t)
        s = np.random.exponential(scale=1 / m)
        ratio = hawkes_intensity(mu, alpha, points, t + s) / m
        if t + s >= et:
            break
        if ratio >= np.random.uniform():
            points.append(t + s)
        all_samples.append(t + s)
        t = t + s
    return points, all_samples


def create_sample(length):
    # https://skill-lync.com/student-projects/Simulation-of-a-Simple-Pendulum-on-Python-95518

    def sim_pen_eq(_, theta):
        dtheta2_dt = (-b / m) * theta[1] + (-g / length) * np.sin(theta[0])
        dtheta1_dt = theta[1]
        return [dtheta1_dt, dtheta2_dt]

    # main

    theta1_ini = np.random.uniform(0, 2 * np.pi)  # Initial angular displacement (rad)
    theta2_ini = np.random.uniform(-np.pi, np.pi)  # Initial angular velocity (rad/s)
    theta_ini = [theta1_ini, theta2_ini]

    et = np.random.uniform(et_min, et_max)
    mu = mean_n * (1 - alpha) / (et - st - 1)

    target = int(np.random.choice(10, 1)[0])
    b = np.linspace(1, 3, 10)[target]  # np.random.uniform(1, 3)

    t_span = [st, et + ts]

    points, _ = simulate_hawkes(mu, alpha, st, et)
    if len(points) < 5:
        points = np.linspace(st, et, 5).tolist()
    t_ir = points

    theta12 = solve_ivp(sim_pen_eq, t_span, theta_ini, t_eval=t_ir)
    theta1 = theta12.y[0, :]

    # return x, y
    # or we could return angles ...
    x = np.sin(theta1)
    y = -np.cos(theta1)

    return x, y, t_ir, target


def noise_sequence(x, y):
    # x += 0. * np.random.randn(x.shape[0])
    # y += 0. * np.random.randn(y.shape[0])
    x[np.random.choice(len(x), size=int(len(x) * 0.1), replace=False)] = np.nan
    y[np.random.choice(len(y), size=int(len(y) * 0.1), replace=False)] = np.nan
    return x, y


def create_dataset(size):

    data = []
    for i in trange(size):
        length = np.random.uniform(0.5, 10)
        x, y, time, target = create_sample(length)
        x, y = noise_sequence(x, y)
        data.append((i, target, x.tolist(), y.tolist(), time, len(time), time[-1]))

    spark = SparkSession.getActiveSession()  # pyright: ignore
    schema = StructType(
        [
            StructField("id", LongType()),
            StructField("target", LongType()),
            StructField("x", ArrayType(FloatType())),
            StructField("y", ArrayType(FloatType())),
            StructField("time", ArrayType(FloatType())),
            StructField("_seq_len", LongType()),
            StructField("_last_time", FloatType()),
        ]
    )
    df = spark.createDataFrame(data, schema)  # pyright: ignore
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--save-path",
        help="Where to save preprocessed parquets",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--seed",
        help="Random seed used to generate data",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        help='Toggle "overwrite" mode on all spark writes',
        action="store_true",
    )
    args = parser.parse_args()
    mode = "overwrite" if args.overwrite else "error"

    np.random.seed(args.seed)
    SparkSession.builder.master("local[32]").getOrCreate()  # pyright: ignore
    args.save_path.mkdir(parents=True, exist_ok=True)

    train_df = create_dataset(train_size)
    test_df = create_dataset(test_size)

    train_df.coalesce(1).write.parquet((args.save_path / "train").as_posix(), mode=mode)
    test_df.coalesce(1).write.parquet((args.save_path / "test").as_posix(), mode=mode)


if __name__ == "__main__":
    main()
