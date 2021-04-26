import os

def get_rank():
    """
    We use environment variables to pass the rank info
    Returns:
        rank: rank in the multi-node system 
        local_rank: local rank on a node
    """
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        rank = 0
        local_rank = 0
        os.environ["RANK"] = str(0)
        os.environ["LOCAL_RANK"] = str(0)
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

    return rank, local_rank