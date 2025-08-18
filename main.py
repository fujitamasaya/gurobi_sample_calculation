# main.py（簡素版：S3指定時のみ出力、ローカル時は出力しない）
import argparse, io, json, time
from pathlib import Path
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--bucket")  # ある時だけS3を使う
    return ap.parse_args()

def load_local_inputs():
    p_csv = Path("input/products.csv")
    p_json = Path("input/capacities.json")
    if not p_csv.exists() or not p_json.exists():
        raise FileNotFoundError("Place files under input/: products.csv, capacities.json")
    df = pd.read_csv(p_csv)
    caps = json.loads(p_json.read_text(encoding="utf-8"))
    return df, caps

def load_s3_inputs(bucket: str, job_id: str):
    import boto3
    s3 = boto3.client("s3")
    prod_key = f"input/{job_id}/products.csv"
    cap_key = f"input/{job_id}/capacities.json"
    get = lambda k: s3.get_object(Bucket=bucket, Key=k)["Body"].read()
    df   = pd.read_csv(io.BytesIO(get(prod_key)))
    caps = json.loads(get(cap_key).decode("utf-8"))
    return df, caps

def write_s3_output(df: pd.DataFrame, bucket: str, job_id: str):
    import boto3
    s3 = boto3.client("s3")
    key = f"output/{job_id}/solution.csv"
    s3.put_object(Bucket=bucket, Key=key, Body=df.to_csv(index=False).encode("utf-8"))
    print(f"[info] wrote s3://{bucket}/{key}")

def solve(products: pd.DataFrame, caps: dict):
    m = gp.Model("pm"); m.Params.OutputFlag = 0
    items  = products["product"].tolist()
    profit = {r.product: float(r.profit) for r in products.itertuples()}
    resA   = {r.product: float(r.resA)   for r in products.itertuples()}
    resB   = {r.product: float(r.resB)   for r in products.itertuples()}

    x = m.addVars(items, lb=0.0, name="x")
    m.setObjective(gp.quicksum(profit[i]*x[i] for i in items), GRB.MAXIMIZE)
    if "resA" in caps: m.addConstr(gp.quicksum(resA[i]*x[i] for i in items) <= float(caps["resA"]))
    if "resB" in caps: m.addConstr(gp.quicksum(resB[i]*x[i] for i in items) <= float(caps["resB"]))
    m.optimize()
    if m.status != GRB.OPTIMAL: raise RuntimeError(f"status={m.status}")
    sol = pd.DataFrame([{"product": i, "quantity": x[i].X, "profit_contrib": profit[i]*x[i].X} for i in items])
    return m.ObjVal, sol

def main():
    a = parse_args()
    use_s3 = bool((a.bucket or "").strip())

    time.sleep(600)

    products, caps = load_s3_inputs(a.bucket, a.job_id) if use_s3 else load_local_inputs()
    obj, sol = solve(products, caps)

    print(f"Objective={obj}")
    print(sol.to_string(index=False))

    # 出力は S3 指定時のみ
    if use_s3:
        write_s3_output(sol, a.bucket, a.job_id)

if __name__ == "__main__":
    main()
