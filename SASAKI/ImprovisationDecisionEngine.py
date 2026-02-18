import yaml
import random

class ImprovisationDecisionEngine:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg["ImprovisationDecision"]
        self.rules = self.cfg["policy"]["rules"]
        self.base_dist = self.cfg["base_distribution"]

    def _select_rule(self, H: float):
        """シャノンエントロピー H に応じて policy.rules から1つ選ぶ。"""
        for rule in self.rules:
            cond = rule["condition"]
            # 超シンプルパーサ（必要最低限）
            if "<=" in cond and ">=" in cond:
                # 例: "0.6 <= H < 1.1"
                left, rest = cond.split("<=")
                _, right = rest.split("<")
                left = float(left.strip())
                right = float(right.strip())
                if left <= H < right:
                    return rule
            elif "<" in cond and "H" in cond:
                # 例: "H < 0.6"
                threshold = float(cond.split("<")[-1])
                if H < threshold:
                    return rule
            elif ">" in cond and "H" in cond:
                # 例: "H >= 1.1"
                threshold = float(cond.split(">=")[-1])
                if H >= threshold:
                    return rule
        # どれにも当てはまらなかったら base_distribution を使う
        return {
            "id": "base",
            "weights": self.base_dist,
            "rationale": "どの条件にも当てはまらなかったため、素の1/3分布を使用。"
        }

    def decide(self, H: float, mode: str = None) -> str:
        """
        H: Genesys/Braceから計算されたシャノンエントロピー。
        mode: "stochastic" or "argmax"（未指定なら YAML の設定を使う）
        戻り値: "advance" / "retreat" / "hold"
        """
        policy_mode = mode or self.cfg["policy"]["mode"]
        rule = self._select_rule(H)
        weights = rule["weights"]

        if policy_mode == "argmax":
            # 最大確率を選ぶ決定論モード
            return max(weights.items(), key=lambda kv: kv[1])[0]

        # デフォルトは確率的サンプリング
        dirs = list(weights.keys())
        probs = [weights[d] for d in dirs]
        s = sum(probs)
        probs = [p / s for p in probs]

        r = random.random()
        acc = 0.0
        for d, p in zip(dirs, probs):
            acc += p
            if r <= acc:
                return d
        return dirs[-1]  # 端数対策

# ===== Brace との接続イメージ =====

def estimate_entropy_from_brace(brace_motion_features) -> float:
    """
    Braceの特徴量から「今どれだけ先が読めないか」をシャノンエントロピーとして推定する関数。
    ここは好きに実装してOK。最初はダミーで定数とかでもいい。
    """
    # 例: brace側が「次に来そうな3パターンの確率」を返してくれると仮定した場合:
    # probs = brace_motion_features["next_phase_probs"]  # [p1, p2, p3]
    # import math
    # H = -sum(p * math.log2(p) for p in probs if p > 0)
    # return H
    return 0.9  # 仮で中くらいのエントロピーを返す

def brace_step(brace_motion_features, engine: ImprovisationDecisionEngine):
    """
    Braceの1ステップ更新の中で、
    ImprovisationDecision をかませるイメージ。
    """
    H = estimate_entropy_from_brace(brace_motion_features)
    direction = engine.decide(H)
    # ここで direction を使って分岐：
    # - "advance" → 攻めのフットワーク／前ノリのポーズ
    # - "retreat" → 距離を取る動き／後ノリ
    # - "hold"    → その場に残る・タメる・フリーズ系 etc.
    return direction
