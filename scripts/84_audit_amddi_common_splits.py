import torch
import os


BASE="data/amddi_common_splits"


def check_overlap(a,b):

    return len(
        set(map(tuple,a.tolist()))
        &
        set(map(tuple,b.tolist()))
    )


for split in [
    "transductive",
    "cold_start_S2"
]:

    print("\n"+"="*70)
    print(split)
    print("="*70)


    path=f"{BASE}/{split}"


    train_pos=torch.load(
        f"{path}/train_pos.pt"
    )

    val_pos=torch.load(
        f"{path}/val_pos.pt"
    )

    test_pos=torch.load(
        f"{path}/test_pos.pt"
    )


    train_neg=torch.load(
        f"{path}/train_neg.pt"
    )


    test_neg=torch.load(
        f"{path}/test_neg.pt"
    )


    print(
        "Train-Val overlap:",
        check_overlap(
            train_pos,
            val_pos
        )
    )

    print(
        "Train-Test overlap:",
        check_overlap(
            train_pos,
            test_pos
        )
    )

    print(
        "Val-Test overlap:",
        check_overlap(
            val_pos,
            test_pos
        )
    )


    print(
        "Positive-Negative overlap:",
        check_overlap(
            train_pos,
            train_neg
        )
    )



    if split=="cold_start_S2":

        train_drugs=set(
            train_pos.flatten().tolist()
        )

        test_drugs=set(
            test_pos.flatten().tolist()
        )


        print(
            "Train-Test drug overlap:",
            len(
                train_drugs & test_drugs
            )
        )
