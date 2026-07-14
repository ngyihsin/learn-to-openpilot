# Loop Report — Round 1(第一輪:考試、批改與檢討)

日期:2026-07-14 · 考題版本:`final_exam.md` @ round 1 · 教材版本:commit `8276412`

## 1. 分數(Scores)

| 學生 | 設定 | 初評 | 嚴格覆核 | ≥90? |
|------|------|------|----------|------|
| S1(程度好) | CS 畢業、細讀全部教材含 Hints/Check-yourself/解答 | 100 | — | ✅ |
| S2(目標客群) | 電機資工畢業、基礎 Python、無 ML 背景、照建議順序讀完 | 100 | 100 | ✅ |
| S3(程度較弱) | Python 生疏、只讀 README 正文與 Hints、跳過 Check-yourself | 100 | 100 | ✅ |

初評三人皆滿分,過於完美反而可疑,因此加開**對抗式嚴格覆核**(獨立 agent、逐 rubric
bullet 檢查、對 hedge 不留情):S2、S3 分數不變 —— 不是批改放水,而是答案內容確實
與教材陳述近乎逐字吻合。詳見 `round1/grading.md`、`round1/regrade_strict.md`。

**分數判準(全員 ≥90)於第一輪即通過。** 但 loop engineering 的重點不只分數 ——
三位學生的應試回報揭露了真正的問題(見下)。

## 2. 錯題/掙扎點檢討(Review)

雖然沒有失分,三位學生**獨立且一致**回報以下題目是「教材沒教、靠推理硬撐」。對真
人學生而言,這些就是預期失分點。逐項分類:**A = 教材缺口、B = 考題瑕疵、C = 學生
個人疏失**。

| # | 訊號(誰回報) | 分類 | 檢討 |
|---|----------------|------|------|
| 1 | **Q18 回歸損失**:全課程從未在影像上訓練過回歸模型 —— 回歸只出現在 Day 33 的手刻直線擬合,Week 4 全是分類 + CrossEntropyLoss(S1、S2、S3 全員) | **A** | 課程目標是「訓練自駕車可用的推論模型」,而方向盤角度、路徑曲率都是連續值 —— 這是最嚴重的缺口 |
| 2 | **Q18 延遲驗證**:課程提過 real-time、fixed rates,但從未示範如何量測推論延遲、frame budget 怎麼算(S1、S2、S3) | **A** | 部署驗證只教了 fidelity(`torch.allclose`),沒教 speed |
| 3 | **Q17(c) 量化三模式**:dynamic/static/QAT 只出現在 Day 28 的 Check-yourself 一行括號裡;S3 這類學生會跳過該區塊(S1、S2、S3) | **A** | 重要事實不能只放在自我檢查題 |
| 4 | **Q15(c)(d) delegate 與 AOT 圖的好處**:只在 Day 27 Check-yourself 出現;S2 说「答案基本上是那一句話撐開的」、S3 承認用猜的 | **A** | 同上 |
| 5 | **Q20(c)「#1 bug」歧義**:Day 22 說 zero_grad 是 #1 bug,Day 28b 說 train-set 評分是 #1 bug,學生只能擲硬幣(S1、S2、S3) | **A**(教材用語衝突)| 兩課的「第一名」需要區分命名 |
| 6 | **Q1 zero_grad 順序不一致**:Day 22 的迴圈把 zero_grad 放最後,Day 23 的提示放最前,S3 不敢確定哪個對 | **A** | 需明講「同一個迴圈的旋轉,重點是下次 backward 前歸零」 |
| 7 | **Q6 數值梯度為何太慢**:理由只在 Day 22 Check-yourself,S3 靠自己推(推對了,真人未必) | **A** | 提進正文/Hints |

**B(考題瑕疵)**:無 —— 嚴格覆核確認每題皆可由教材作答。
**C(學生疏失)**:無失分,無此類。

## 3. 本輪改進(Fixes applied)

| 對應 # | 修正 |
|--------|------|
| 1, 2 | **新增 Day 28c「Steering-Angle Regression: The Driving Mini-Project」**(README + homework + solution + 自動評分器,已驗證 `LP_IMPL=solution pytest` 6/6 通過):合成道路影像 → CNN 回歸連續轉向值 → `nn.MSELoss`(並說明 `nn.SmoothL1Loss`/Huber 的使用時機)→ 以 **val MAE** 選最佳 checkpoint → `measure_latency_ms`(warmup + median)→ `meets_frame_budget`(20 Hz → 50 ms)。同步更新 SYLLABUS、Week 4 README、Day 28/28b 的銜接 |
| 3 | Day 28 正文新增量化三模式段落(dynamic/static/QAT、QAT 恢復最多精度),學習目標同步 |
| 4 | Day 27 正文新增 AOT 圖的三個好處(無直譯器、AOT 最佳化、固定記憶體)與 delegate 的完整說明,學習目標同步 |
| 5 | Day 28b 改寫為「Day 22 的 #1 bug 是忘記歸零梯度;**評估**的 #1 bug 是拿訓練集自評」 |
| 6 | Day 22 Hints 新增「zero_grad 放最前(Day 23 寫法)是同一迴圈的旋轉」說明 |
| 7 | Day 22 Hints 新增數值梯度成本分析(每參數兩次 forward vs. 一次 backward 全拿) |

## 4. 模擬方法的誠實聲明(Known limitation)

LLM 模擬學生帶有潛在的專家知識,即使 persona 限制「只能用教材內容」,推理能力仍強
於同背景真人 —— 這使分數偏樂觀(本輪全員 100)。因此本迴圈把**質性訊號**(學生自
述的掙扎點)視為與分數同等重要的檢討輸入;上表 7 項缺口全部來自該訊號。對真人學
生,建議以本考卷實測並將結果回饋到下一輪。

## 5. 下一輪(Round 2)

教材已更新,Round 2 重考 S2(目標客群)與 S3(最弱、鑑別力最高),研讀清單加入
Day 28c,驗證:(a) 分數維持 ≥90;(b) Round 1 回報的掙扎點消失(特別是 Q17c、Q15、
Q18 的損失函數與延遲部分)。S1 於 Round 1 已滿分且教材只增不減,成績沿用。
