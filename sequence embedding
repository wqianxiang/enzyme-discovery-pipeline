print("--------------------------------------------------------------------")
print("This was created by Qianxiang😊, from HKUST-GZ🏫")
print("I would recommend you to use this pipleine in WSL or Linux system")
print("--------------------------------------------------------------------")

import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
import math
import argparse
import sys
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, set_link_color_palette
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
import glob
import time

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'Arial'

# 清洗蛋白质序列
def process_fasta(file):
    data = []
    filtered_number = 0
    with open(file, 'r') as f:
        file_content = f.read().split('>')
        for lines in file_content:
            if not lines:
                continue
            description_sequence = lines.strip().split('\n')[0]
            name = description_sequence.strip().split()[0]
            sequence = ''.join(lines.strip().split('\n')[1:])
            if sequence and all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
                data.append((name,sequence))
    return data

#对序列嵌入向量
def get_embedding(data, batch_size, results):

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    sequence_embedding = []
    sequence_name = []
    sequence = []
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("/root/autodl-tmp/model_depolyment/protein_LLM/ProstT5", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/root/autodl-tmp/model_depolyment/protein_LLM/ProstT5").to(device)
    model.float() if device.type == "cpu" else model.half()
    model.eval()

    #总数据量
    total_sequence = len(data)
    #批次数量
    batch_number = math.ceil(len(data)/batch_size)

    #假设每个batch有5条序列则从0开始，start_idx=0,end_idx=5
    for i in range(batch_number):
        print(f"正在处理第 {i + 1}/{batch_number} 个批次")
        start_idx = i*batch_size
        end_idx = (i+1)*batch_size
        if end_idx > total_sequence:
            end_idx = total_sequence

        sequence_str = []
        model_input = []

        for i in range(start_idx,end_idx):
            sequence_str.append(data[i][1])

        for s in sequence_str:
            if s.isupper():
                model_input.append("<AA2fold>" + " " +" ".join(list(s)))
            else:
                model_input.append("<fold2AA>" + " " +" ".join(list(s)))
        #将序列分词并填充至批次中最长的序列。
        ids = tokenizer.batch_encode_plus(model_input, 
                                          add_special_tokens=True, 
                                          padding="longest",
                                          return_tensors='pt').to(device)

        #模型推理，输出的结果为：token_representations.last_hidden_state    
        with torch.no_grad():
            token_representations = model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )

        #模型给每一个氨基酸一个向量，这里将所有氨基酸向量进行平均，得到一条序列的向量
        for i in range(len(sequence_str)):
            sequence_embedding.append(token_representations.last_hidden_state[i,1:len(sequence_str[i])+1].mean(dim=0).cpu().numpy())
            sequence_name.append(data[start_idx+i][0])
            sequence.append(sequence_str[i])

    results = {
    "sequence_embedding": sequence_embedding,
    "sequence_name": sequence_name,
    "sequence": sequence,}

    #释放显存
    del sequence_embedding, sequence_name, model_input, token_representations
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def cluster(results):
    data = np.load(results)
    sequence_embedding = np.array(data['sequence_embedding'])
    sequence_name = data['sequence_name']
    sequence = data['sequence']
    distance = pdist(sequence_embedding, metric='euclidean')
    linkage_matrix = linkage(distance, method='ward', metric='euclidean')
    clusters = fcluster(linkage_matrix, t=4, criterion='distance')
    return clusters, linkage_matrix, sequence_name, sequence

def save_cluster_ids(clusters, sequence_name, sequence, output_path="cluster_results.csv"):
    data = {
        'Cluster': clusters,
        'Sequence_Name': sequence_name,
        'Sequences': sequence
        }
    df = pd.DataFrame(data)
    df = df.sort_values(by='Cluster')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'cluster result has been saved in {output_path}')

def plot_circular_cluster(linkage_matrix, color_threshold=4, show_labels=False, label_every=50):
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['lines.linewidth'] = 1

    cluster_colors = [
        '#4A90E2',
        '#D0021B',
        '#50E3C2',
        '#F5A623',
        '#7B8B99',
        '#B8E986',
        '#8B572A'
    ]

    set_link_color_palette(cluster_colors)

    cluster = dendrogram(
        linkage_matrix,
        color_threshold=color_threshold,
        no_plot=True,
        no_labels=True,
        above_threshold_color='#000000'
    )

    n_leaves = len(cluster['leaves'])
    max_distance = max(max(d) for d in cluster['dcoord'])

    inner_radius = 0.08
    outer_radius = 1.0

    def to_polar(x, y):
        theta = 2 * np.pi * np.asarray(x) / (10 * n_leaves)
        radius = inner_radius + (outer_radius - inner_radius) * (1 - np.asarray(y) / max_distance)
        return theta, radius

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_axis_off()

    # 画圆形树枝
    for xs, ys, color in zip(cluster['icoord'], cluster['dcoord'], cluster['color_list']):
        for i in range(3):
            x_segment = np.linspace(xs[i], xs[i + 1], 40)
            y_segment = np.linspace(ys[i], ys[i + 1], 40)

            theta, radius = to_polar(x_segment, y_segment)
            ax.plot(theta, radius, color=color, linewidth=1)

    # 外圈颜色带
    leaf_colors = cluster.get('leaves_color_list', None)
    if leaf_colors is not None:
        width = 2 * np.pi / n_leaves
        theta = np.arange(n_leaves) * width

        ax.bar(
            theta,
            height=0.045,
            width=width,
            bottom=outer_radius + 0.015,
            color=leaf_colors,
            edgecolor='none',
            align='edge'
        )

    if show_labels:
        for i, leaf_id in enumerate(cluster['leaves']):
            if i % label_every != 0:
                continue

            theta = 2 * np.pi * (5 + 10 * i) / (10 * n_leaves)
            angle = (90 - np.degrees(theta)) % 360

            if 90 < angle < 270:
                rotation = angle + 180
                ha = 'right'
            else:
                rotation = angle
                ha = 'left'

            ax.text(
                theta,
                outer_radius + 0.09,
                str(leaf_id),
                fontsize=8,
                rotation=rotation,
                rotation_mode='anchor',
                ha=ha,
                va='center'
            )

    # 图例
    actual_colors = set(cluster['color_list'])
    discovered_colors = [c for c in cluster_colors if c in actual_colors]

    legend_handles = [
        mpatches.Patch(color=color, label=f'Cluster {i + 1}')
        for i, color in enumerate(discovered_colors)
    ]

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc='upper right',
            bbox_to_anchor=(1.15, 1.10),
            fontsize=10,
            frameon=False
        )

    plt.tight_layout()
    plt.savefig('cluster_circular_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.savefig('cluster_circular_dendrogram.pdf', bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="sequence embedding module")
    parser.add_argument('--file', dest='file', required=True, help='input fasta file path')
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='batch size for sequence embedding')
    parser.add_argument('--embedding_output', required=False, default='sequence_embedding.npz', help='output npz file path for sequence embedding results')
    parser.add_argument('--cluster_output', required=False, default='cluster_results.csv', help='output csv file path for cluster results')
    args = parser.parse_args()

    user_input = input('Do you want to start the sequence embedding module of EnzyQXminer (yes/no):').strip().lower()
    if user_input =='yes':
        if glob.glob('sequence_embedding.npz'):

            print('step3🚀: Clustering your sequences')
            time.sleep(1)
            clusters, linkage_matrix, sequence_name, sequence = cluster(args.embedding_output)

            print('step4🚀: Saving cluster result')
            time.sleep(1)
            save_cluster_ids(clusters, sequence_name, sequence, args.cluster_output)

            print('step5🚀: Plotting cluster result')
            time.sleep(1)
            plot_cluster(linkage_matrix)
        else:
            print('step1🚀: Processing you input fasta file')
            time.sleep(1)
            data = []
            data = process_fasta(args.file)

            if len(data) == 0:
                print('No valid sequence in your input fasta file')
                sys.exit(1)

            print('step2🚀: Computing the embedding of your sequences')
            time.sleep(1)
            results = []
            results = get_embedding(data, args.batch_size, results)

            if len(results) == 0:
                print('No embedding result was generated')
                sys.exit(1)

            np.savez(args.embedding_output,
                    sequence_embedding=results['sequence_embedding'],
                    sequence_name=results['sequence_name'],
                    sequence=results['sequence'])

            print('step3🚀: Clustering your sequences')
            time.sleep(1)
            clusters, linkage_matrix, sequence_name, sequence = cluster(args.embedding_output)

            print('step4🚀: Saving cluster result')
            time.sleep(1)
            save_cluster_ids(clusters, sequence_name, sequence, args.cluster_output)

            print('step5🚀: Plotting cluster result')
            time.sleep(1)
            plot_cluster(linkage_matrix)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

