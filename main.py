# main.py（S3指定時のみ {job_id}/logs/ にログ出力）
import argparse, io, json, time, sys, traceback, datetime
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
    prod_key = f"{job_id}/input/products.csv"
    cap_key = f"{job_id}/input/capacities.json"
    get = lambda k: s3.get_object(Bucket=bucket, Key=k)["Body"].read()
    df   = pd.read_csv(io.BytesIO(get(prod_key)))
    caps = json.loads(get(cap_key).decode("utf-8"))
    log("[info] loaded inputs from S3", {"bucket": bucket, "prod_key": prod_key, "cap_key": cap_key})
    return df, caps

def write_s3_output(df: pd.DataFrame, bucket: str, job_id: str):
    import boto3
    s3 = boto3.client("s3")
    key = f"{job_id}/output/solution.csv"
    s3.put_object(Bucket=bucket, Key=key, Body=df.to_csv(index=False).encode("utf-8"))
    log("[info] wrote solution", {"s3_uri": f"s3://{bucket}/{key}"})

# --- ロギング周り ------------------------------------------------------------
_APP_LOG_PATH = "/tmp/app.log"
_GRB_LOG_PATH = "/tmp/gurobi.log"

def log(msg: str, payload: dict | None = None):
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    if payload:
        print(f"{ts} {msg} {json.dumps(payload, ensure_ascii=False)}")
    else:
        print(f"{ts} {msg}")
    # 同時にファイルにも追記
    with open(_APP_LOG_PATH, "a", encoding="utf-8") as f:
        if payload:
            f.write(f"{ts} {msg} {json.dumps(payload, ensure_ascii=False)}\n")
        else:
            f.write(f"{ts} {msg}\n")

def upload_logs_if_needed(use_s3: bool, bucket: str | None, job_id: str):
    if not use_s3:
        return
    import boto3, os
    s3 = boto3.client("s3")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    app_key = f"{job_id}/logs/app_{ts}.log"
    grb_key = f"{job_id}/logs/gurobi_{ts}.log"
    # app.log
    if Path(_APP_LOG_PATH).exists():
        s3.upload_file(_APP_LOG_PATH, bucket, app_key)
        print(f"[info] uploaded app log to s3://{bucket}/{app_key}")
    # gurobi.log
    if Path(_GRB_LOG_PATH).exists():
        s3.upload_file(_GRB_LOG_PATH, bucket, grb_key)
        print(f"[info] uploaded gurobi log to s3://{bucket}/{grb_key}")


# --- 起動時ログ（最小）＆ライセンスログ初期化 -------------------------------
def log_startup(job_id: str, use_s3: bool):
    py_ver = sys.version.split()[0]
    grb_py_ver = getattr(gp, "__version__", "unknown")
    log("[info] startup", {
        "job_id": job_id, "use_s3": use_s3,
        "python": py_ver, "platform": platform.platform(),
        "gurobipy": grb_py_ver
    })

def init_gurobi_logging():
    """最小：ライセンス解決を走らせつつ、ログファイルだけ確実に出させる"""
    try:
        env = gp.Env(empty=True)
        env.setParam("LogFile", _GRB_LOG_PATH)  # ライセンス情報含む全ログを /tmp/gurobi.log へ
        env.start()
        env.dispose()
        log("[info] gurobi log initialized", {"path": _GRB_LOG_PATH})
    except Exception as e:
        log("[warn] gurobi log init failed", {"error": str(e)})

# --- 最適化 -------------------------------------------------------------------
def solve(products: pd.DataFrame, caps: dict, enable_solver_log: bool):
    m = gp.Model("pm")
    # コンソール出力は抑えつつ、S3出力時のみファイルに Gurobi ログ保存
    m.Params.OutputFlag = 0
    if enable_solver_log:
        m.Params.LogFile = _GRB_LOG_PATH  # /tmp にログ生成（後でS3へアップ）

    items  = products["product"].tolist()
    profit = {r.product: float(r.profit) for r in products.itertuples()}
    resA   = {r.product: float(r.resA)   for r in products.itertuples()}
    resB   = {r.product: float(r.resB)   for r in products.itertuples()}

    x = m.addVars(items, lb=0.0, name="x")
    m.setObjective(gp.quicksum(profit[i]*x[i] for i in items), GRB.MAXIMIZE)
    if "resA" in caps: m.addConstr(gp.quicksum(resA[i]*x[i] for i in items) <= float(caps["resA"]))
    if "resB" in caps: m.addConstr(gp.quicksum(resB[i]*x[i] for i in items) <= float(caps["resB"]))

    log("[info] start optimize", {"n_var": len(items), "caps": caps})
    m.optimize()
    log("[info] optimize finished", {"status": int(m.status), "obj": float(m.ObjVal) if m.Status == GRB.OPTIMAL else None})

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"status={m.status}")

    sol = pd.DataFrame([
        {"product": i, "quantity": x[i].X, "profit_contrib": profit[i]*x[i].X}
        for i in items
    ])
    return m.ObjVal, sol

# --- main --------------------------------------------------------------------
def main():
    a = parse_args()
    use_s3 = bool((a.bucket or "").strip())

    # 起動ログ（簡素）
    log_startup(a.job_id, use_s3)
    init_gurobi_logging()
    log("[info] job start", {"job_id": a.job_id, "use_s3": use_s3})

    try:
        # デモ用の待機
        log("[info] sleeping", {"seconds": 600})
        time.sleep(600)

        products, caps = load_s3_inputs(a.bucket, a.job_id) if use_s3 else load_local_inputs()
        obj, sol = solve(products, caps, enable_solver_log=use_s3)

        log("[info] objective", {"value": float(obj)})
        log("[info] solution_preview", {"rows": len(sol)})

        # 画面にも表示（開発用）
        print(f"Objective={obj}")
        print(sol.to_string(index=False))

        if use_s3:
            write_s3_output(sol, a.bucket, a.job_id)

        log("[info] job success", {"job_id": a.job_id})
    except Exception as e:
        # 例外内容もログに残す
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        log("[error] job failed", {"error": str(e)})
        log("[error] traceback", {"traceback": tb})
        raise
    finally:
        # S3 指定時のみログをアップロード
        upload_logs_if_needed(use_s3, a.bucket, a.job_id)

if __name__ == "__main__":
    main()
