import torch
import random
import os

from itertools import combinations


SEED=42

random.seed(SEED)


OUT="data/amddi_common_splits"


os.makedirs(
    OUT,
    exist_ok=True
)



# ==========================================
# Load DDI edges
# ==========================================

edge=torch.load(
    "graphs/amddi_common/ddi_edge_index.pt"
)


# remove reverse duplicates

positive=set()


for a,b in edge.t().tolist():

    if a<b:

        positive.add(
            (a,b)
        )


positive=list(positive)


print(
    "Positive DDI pairs:",
    len(positive)
)



num_drugs=1087



# ==========================================
# Negative pool
# ==========================================

all_pairs=set(
    combinations(
        range(num_drugs),
        2
    )
)


negative_pool=list(
    all_pairs-set(positive)
)


random.shuffle(
    negative_pool
)


negative_pool=negative_pool[
    :len(positive)
]


print(
    "Negative pairs:",
    len(negative_pool)
)



def save_split(
    name,
    train_pos,
    val_pos,
    test_pos
):

    path=f"{OUT}/{name}"

    os.makedirs(
        path,
        exist_ok=True
    )


    neg=random.sample(
        negative_pool,
        len(train_pos)+len(val_pos)+len(test_pos)
    )


    train_neg=neg[
        :len(train_pos)
    ]

    val_neg=neg[
        len(train_pos):
        len(train_pos)+len(val_pos)
    ]

    test_neg=neg[
        len(train_pos)+len(val_pos):
    ]


    for fname,data in {

        "train_pos.pt":train_pos,
        "val_pos.pt":val_pos,
        "test_pos.pt":test_pos,
        "train_neg.pt":train_neg,
        "val_neg.pt":val_neg,
        "test_neg.pt":test_neg

    }.items():

        torch.save(
            torch.tensor(data),
            f"{path}/{fname}"
        )


    print(
        name,
        len(train_pos),
        len(val_pos),
        len(test_pos)
    )



# ==========================================
# Transductive
# ==========================================

random.shuffle(
    positive
)


n=len(positive)

train=int(0.8*n)
val=int(0.1*n)


save_split(
    "transductive",
    positive[:train],
    positive[train:train+val],
    positive[train+val:]
)



# ==========================================
# Cold start S2
# drug-disjoint
# ==========================================

drugs=list(
    range(num_drugs)
)

random.shuffle(drugs)


cut=int(
    0.8*num_drugs
)


train_drugs=set(
    drugs[:cut]
)


test_drugs=set(
    drugs[cut:]
)


train=[]
test=[]


for e in positive:

    a,b=e

    if (
        a in train_drugs
        and
        b in train_drugs
    ):

        train.append(e)


    elif (
        a in test_drugs
        and
        b in test_drugs
    ):

        test.append(e)



val_size=int(
    0.1*len(train)
)


save_split(
    "cold_start_S2",
    train[val_size:],
    train[:val_size],
    test
)


print("DONE")
