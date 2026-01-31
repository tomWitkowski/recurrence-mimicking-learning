import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def set_plot_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })


def generate_ar1_price(T, rng, spread=0.0004):
    ret_raw = np.zeros(T)
    for t in range(1, T):
        ret_raw[t] = 0.8 * ret_raw[t - 1] + rng.normal(0, 0.1)
    price = 100 + np.cumsum(ret_raw)
    bid = price - spread / 2
    ask = price + spread / 2
    ba = np.stack([bid, ask], axis=1).astype(np.float32)
    return price, ba


def make_features_from_price(price, window=50):
    dprice = np.concatenate([[0.0], np.diff(price)])
    x = np.zeros((len(price), window), dtype=np.float32)
    for i in range(len(price)):
        start = max(0, i - (window - 1))
        w = dprice[start:i + 1]
        x[i, -len(w):] = w
    return x


def generate_trend_series(T, rng):
    trend = np.zeros(T, dtype=np.int8)
    t = 0
    prev = 1
    while t < T:
        seg_len = int(rng.integers(200, 600))
        if rng.random() < 0.2:
            cur = 0
        else:
            cur = -prev if rng.random() < 0.7 else prev
        end = min(T, t + seg_len)
        trend[t:end] = cur
        if cur != 0:
            prev = cur
        t = end

    base_slope = trend * rng.uniform(0.03, 0.07, size=T)
    trend_noise = np.convolve(
        rng.normal(0, 0.05, size=T),
        np.ones(51, dtype=np.float32) / 51,
        mode='same',
    )
    noise = rng.normal(0, 0.2, size=T)
    ret = base_slope + trend_noise + noise
    price = 100 + np.cumsum(ret)

    k = int(T * 0.002)
    idx = rng.choice(T, size=k, replace=False)
    sign = rng.choice([-1.0, 1.0], size=k)
    price[idx] += 4.0 * sign

    label = trend.astype(np.int8)
    label[idx] = (2 * sign).astype(np.int8)

    return price, label


def plot_trading_actions(price, actions, actions_raw):
    actions_np = actions.numpy().reshape(-1)
    actions_disc = np.clip(np.round(actions_np), -1, 1).astype(int)

    price_np = np.asarray(price).reshape(-1)
    n = min(len(price_np), len(actions_disc))
    price_np = price_np[:n]
    actions_disc = actions_disc[:n]

    x = np.arange(n)
    points = np.array([x, price_np]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_actions = actions_disc[:-1]

    color_map = {1: '#2ca02c', 0: '#7f7f7f', -1: '#d62728'}
    colors = [color_map.get(int(a), '#7f7f7f') for a in seg_actions]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set_facecolor('#f7f7f7')

    lc = LineCollection(segments, colors=colors, linewidths=2.2, alpha=0.9)
    ax.add_collection(lc)

    ax.scatter(
        x,
        price_np,
        s=10,
        c=[color_map.get(int(a), '#7f7f7f') for a in actions_disc],
        alpha=0.6,
        linewidths=0,
    )

    ax.set_xlim(x.min(), x.max())
    pad = (price_np.max() - price_np.min()) * 0.08 if n > 1 else 1.0
    ax.set_ylim(price_np.min() - pad, price_np.max() + pad)

    ax.grid(True, alpha=0.25)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax2 = ax.twinx()
    ax2.set_ylim(-1, 1)
    ax2.plot(
        actions_raw.numpy().reshape(-1)[:n],
        label='raw actions',
        color='#1f77b4',
        alpha=0.5,
    )
    ax2.set_ylabel('Raw action')

    legend_items = [
        Line2D([0], [0], color=color_map[1], lw=3, label='Buy (1)'),
        Line2D([0], [0], color=color_map[0], lw=3, label='Hold (0)'),
        Line2D([0], [0], color=color_map[-1], lw=3, label='Sell (-1)'),
        Line2D([0], [0], color='#1f77b4', lw=2, label='Raw actions'),
    ]
    ax.legend(handles=legend_items, frameon=False, loc='upper left')
    ax.set_title('Trading: price colored by actions')
    ax.set_xlabel('Step')
    ax.set_ylabel('Price')
    plt.show()


def plot_trend_with_price(y_true, y_pred, price, n=2000):
    x_idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set_facecolor('#f7f7f7')

    ax.plot(
        x_idx,
        y_true[:n],
        label='true label',
        color='#7f7f7f',
        lw=1.6,
    )
    ax.plot(
        x_idx,
        y_pred[:n],
        label='pred label',
        color='#2ca02c',
        lw=1.2,
        alpha=0.8,
    )

    ax.set_xlim(0, n - 1)
    ax.set_ylim(-2.4, 2.4)
    ax.grid(True, alpha=0.25)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(
        x_idx,
        price[:n],
        label='price',
        color='#1f77b4',
        alpha=0.6,
        lw=1.2,
    )
    ax2.set_ylabel('Price')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper left')
    ax.set_title('Trend labels vs prediction (first 2k)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Trend label')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    classes = np.asarray(classes).astype(np.int32)
    idx_map = {int(v): i for i, v in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[idx_map[int(yt)], idx_map[int(yp)]] += 1
    acc = float((y_pred == y_true).mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix (acc={acc:.3f})')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=10)
    ax.set_facecolor('#f7f7f7')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.show()
    return acc


def plot_regression(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y_true, label='true', alpha=0.6)
    ax.plot(y_pred, label='pred', alpha=0.7, ls='--')
    ax.legend(frameon=False)
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.25)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.show()
