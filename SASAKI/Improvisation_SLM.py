from dataclasses import dataclass, asdict
from enum import Enum
import math
import random
from typing import List, Dict, Optional


class State(str, Enum):
    F = "F"  # 前進
    B = "B"  # 後退
    S = "S"  # 停滞


@dataclass
class SLMConfig:
    # 自由パラメータ6個（残りは 1 - (p_xy + p_xz)）
    p_ff: float = 0.6
    p_fb: float = 0.2
    p_bf: float = 0.3
    p_bb: float = 0.5
    p_sf: float = 0.4
    p_sb: float = 0.4

    temperature_tau: float = 1.0    # ゆらぎ量
    skill_level: float = 0.5       # 停滞のコントロール力
    learning_rate_eta: float = 0.1 # 学習率

    def transition_matrix(self):
        """
        3x3行列 P[state_from][state_to]
        行の残りは 1 - (p_xy + p_xz) で S行きにする。
        順番は [F, B, S]
        """
        # F行
        p_fs = max(0.0, 1.0 - (self.p_ff + self.p_fb))
        # B行
        p_bs = max(0.0, 1.0 - (self.p_bf + self.p_bb))
        # S行
        p_ss = max(0.0, 1.0 - (self.p_sf + self.p_sb))

        return {
            State.F: {State.F: self.p_ff, State.B: self.p_fb, State.S: p_fs},
            State.B: {State.F: self.p_bf, State.B: self.p_bb, State.S: p_bs},
            State.S: {State.F: self.p_sf, State.B: self.p_sb, State.S: p_ss},
        }


@dataclass
class SLMState:
    current_state: State = State.F
    internal_clock: float = 0.0
    last_entropy: float = 0.0
    step_index: int = 0


@dataclass
class LogEntry:
    t: int
    state_from: State
    state_to: State
    entropy: float
    reward: float
    note: str = ""


class ImprovisationSLM:
    """
    3状態マルコフ連鎖 + 温度パラメータ + 熟練度ゲート で構成された
    必要最小限の即興エンジン。
    """

    def __init__(self, config: SLMConfig):
        self.config = config
        self.log: List[LogEntry] = []

    @staticmethod
    def softmax(logits, tau: float):
        if tau <= 0:
            # τ→0 なら argmax に近づける
            max_idx = max(range(len(logits)), key=lambda i: logits[i])
            out = [0.0] * len(logits)
            out[max_idx] = 1.0
            return out

        exps = [math.exp(l / tau) for l in logits]
        s = sum(exps)
        if s == 0:
            return [1.0 / len(logits)] * len(logits)
        return [e / s for e in exps]

    def compute_context_entropy(
        self,
        audio_features,
        pose_features,
    ) -> float:
        """
        シャノンエントロピーの超ざっくり版。
        実際には BRACE の音特徴・ポーズ変化量から確率分布を作って
        H = - Σ p log p を計算するイメージ。
        ここでは placeholder 実装。
        """
        # 例: 正規化エネルギーっぽいものをまとめて 0〜1 に押し込む
        val = float(abs(audio_features)) + float(abs(pose_features))
        # 適当にスケーリングして 0〜1 くらいに
        return max(0.0, min(1.0, val / 10.0))

    def step(
        self,
        state: SLMState,
        audio_entropy_source,
        pose_entropy_source,
        reward: float = 0.0,
    ) -> SLMState:
        """
        SLM を1ステップ進める。
        - audio_entropy_source / pose_entropy_source は
          BRACEから切ったウィンドウの特徴量から計算した指標を想定。
        """
        cfg = self.config
        P = cfg.transition_matrix()

        # 文脈エントロピー（本当はここでシャノンエントロピーを計算）
        context_entropy = self.compute_context_entropy(
            audio_entropy_source, pose_entropy_source
        )

        # 温度をエントロピーで少し動かす例
        # エントロピーが高いほどバラける
        tau = cfg.temperature_tau * (0.5 + context_entropy)

        # 現状態の遷移確率を logits にして softmax
        row = P[state.current_state]
        states = [State.F, State.B, State.S]
        base_probs = [row[s] for s in states]
        # logを取ることで softmax の入力に
        logits = [math.log(max(p, 1e-8)) for p in base_probs]

        probs = self.softmax(logits, tau)

        # 熟練度で S 行きの確率をゲート
        # skill=0 → Sはほぼ出ない
        # skill=1 → Sの確率そのまま
        idx_S = states.index(State.S)
        probs[idx_S] *= cfg.skill_level

        # 正規化し直す
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        else:
            probs = [1.0 / len(probs)] * len(probs)

        # サンプリング
        r = random.random()
        cum = 0.0
        next_state = states[-1]
        for s_state, p in zip(states, probs):
            cum += p
            if r <= cum:
                next_state = s_state
                break

        # ログ
        self.log.append(
            LogEntry(
                t=state.step_index,
                state_from=state.current_state,
                state_to=next_state,
                entropy=context_entropy,
                reward=reward,
            )
        )

        # 学習（超シンプルな報酬付き微調整；本格的なRLにするならここ拡張）
        if reward != 0.0:
            # 報酬が正 → 選んだ遷移の確率を少し上げる、負 → 下げる
            eta = cfg.learning_rate_eta
            from_row = P[state.current_state]
            old_p = from_row[next_state]
            delta = eta * reward
            new_p = max(0.0, min(1.0, old_p + delta))
            # 簡易的に、選んだ遷移だけ書き換えて normalize は省略
            if state.current_state == State.F and next_state == State.F:
                cfg.p_ff = new_p
            elif state.current_state == State.F and next_state == State.B:
                cfg.p_fb = new_p
            elif state.current_state == State.B and next_state == State.F:
                cfg.p_bf = new_p
            elif state.current_state == State.B and next_state == State.B:
                cfg.p_bb = new_p
            elif state.current_state == State.S and next_state == State.F:
                cfg.p_sf = new_p
            elif state.current_state == State.S and next_state == State.B:
                cfg.p_sb = new_p

        # 状態更新
        return SLMState(
            current_state=next_state,
            internal_clock=(state.internal_clock + 0.25) % 1.0,
            last_entropy=context_entropy,
            step_index=state.step_index + 1,
        )
