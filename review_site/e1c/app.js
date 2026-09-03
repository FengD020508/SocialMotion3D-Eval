(function () {
  "use strict";

  const SCHEMA_VERSION = "e1c-human-review-2.0";
  const STORAGE_KEY = "socialmotion3d-e1c-human-review-v2";
  const LANGUAGE_KEY = "socialmotion3d-e1c-language";
  const CLIPS = [
    "01_gp_set_0003_vid_0004_gp_3467_Other_blind.mp4",
    "02_gp_set_0003_vid_0005_gp_6979_HG_blind.mp4",
    "03_gp_set_0001_vid_0005_gp_936_HG_blind.mp4",
    "04_gp_set_0003_vid_0004_gp_2883_HG_blind.mp4",
    "05_gp_set_0008_vid_0001_gp_5803_HG_blind.mp4",
    "06_gp_set_0003_vid_0004_gp_3598_HG_day_high_blind.mp4",
    "07_gp_set_0003_vid_0004_gp_3052_HG_day_high_blind.mp4",
    "08_gp_set_0003_vid_0004_gp_3526_HEAD_day_low_blind.mp4",
    "09_gp_set_0003_vid_0004_gp_2775_HEAD_day_medium_blind.mp4",
    "10_gp_set_0009_vid_0001_gp_6877_HEAD_day_high_blind.mp4",
    "11_gp_set_0003_vid_0005_gp_6938_HEAD_night_high_blind.mp4",
    "12_gp_set_0005_vid_0005_gp_5851_HEAD_night_medium_blind.mp4",
    "13_gp_set_0008_vid_0004_gp_6517_HEAD_night_high_blind.mp4",
    "14_gp_set_0008_vid_0005_gp_6158_HEAD_night_medium_blind.mp4",
    "15_gp_set_0003_vid_0005_gp_6945_HEAD_night_low_blind.mp4",
    "16_gp_set_0008_vid_0001_gp_5931_HG_night_medium_blind.mp4",
    "17_gp_set_0003_vid_0001_gp_3284_STATIC_day_high_blind.mp4",
    "18_gp_set_0005_vid_0005_gp_5860_STATIC_night_low_blind.mp4",
    "19_gp_set_0003_vid_0004_gp_2869_LOS_day_low_blind.mp4",
    "20_gp_set_0003_vid_0004_gp_3555_FTT_day_low_blind.mp4",
    "21_gp_set_0003_vid_0004_gp_2727_CROSS_INIT_LOS_day_medium_blind.mp4",
    "22_gp_set_0005_vid_0003_gp_4043_CROSS_INIT_OCC_day_high_blind.mp4",
    "23_gp_set_0005_vid_0005_gp_5751_OCCLUSION_NIGHT_night_high_blind.mp4",
    "24_gp_set_0003_vid_0004_gp_3394_WTT_LOS_day_medium_blind.mp4",
    "26_gp_set_0008_vid_0005_gp_6137_HG_NIGHT_night_high_blind.mp4",
    "39_gp_set_0001_vid_0003_gp_250_HG_NIGHT_CROSS_unknown_medium_blind.mp4",
    "40_gp_set_0002_vid_0005_gp_3759_HG_LANE_CROSS_unknown_high_blind.mp4",
    "42_gp_set_0005_vid_0002_gp_4199_HG_LONG_LOS_day_low_blind.mp4"
  ];

  const ITEMS = [
    ...CLIPS.map((file, index) => ({ id: `shared-${index + 1}`, type: "shared", file, sectionIndex: index })),
    ...CLIPS.map((file, index) => ({ id: `desync-${index + 1}`, type: "desync", file, sectionIndex: index }))
  ];

  const COPY = {
    zh: {
      pageTitle: "SocialMotion3D · E1c 人工盲评",
      brandSubtitle: "E1c 人工盲评",
      notSaved: "尚未保存",
      autosaved: "已自动保存",
      exportReminder: "请及时导出备份",
      exportCurrent: "导出当前结果",
      heroTitle: "判断动作是否自然地<br>发生在这条轨迹上",
      heroCopy: "全程约 25–35 分钟。结果只保存在当前浏览器；完成后请导出 CSV，并发回实验组织者。",
      motionComparison: "动作方法比较",
      coherenceCheck: "动作—轨迹协调检查",
      totalVideos: "总视频数",
      beforeStart: "开始前",
      introRule1: "请独立评分，不与其他评审讨论，也不要寻找或打开 <code>blind_key.json</code>。",
      introRule2: "视频可重复播放。A/B 均为匿名结果；相近时请选择“一致且都可接受”，不要强迫猜测。",
      introRule3: "球形头部不提供视线方向。不要据此评价精细的头部朝向；看不清时请使用“无法评价”。",
      reviewerId: "评审代号",
      reviewerPlaceholder: "例如 R01（请勿填写真实姓名）",
      reviewerHelp: "代号仅用于区分多位评审的导出文件。",
      startReview: "开始评分",
      resumeReview: "继续 {reviewer} 的评分（{count} / {total}）",
      focusOn: "评价时请关注",
      partOne: "第一部分",
      partTwo: "第二部分",
      sampleCount: "样本 {current} / {total}",
      sharedTitle: "共享轨迹上的动作比较",
      sharedInstruction: "A/B 使用完全相同的根轨迹。请分别判断动作本身的质量，以及对左侧参考中可见交互的保留程度。",
      sharedLegend: "先看<strong>整体动作是否自然可信</strong>，再看挥手、观察、起步或避让等<strong>可见交互是否被保留</strong>。两项判断不要混为一项。",
      desyncTitle: "动作—轨迹协调性比较",
      desyncInstruction: "请把 A/B 视为两种匿名动作—轨迹组合，只判断哪一侧的整体配合更自然。",
      desyncLegend: "可以留意步态与位移、支撑脚与身体推进、起步和停步是否协调；不需要先判断两侧是否存在差异。",
      videoFallback: "当前浏览器无法播放该视频，请改用最新版 Edge 或 Chrome。",
      videoLeft: "左：原始参考视频",
      videoAB: "中 / 右：匿名结果 A / B",
      videoControl: "可暂停、拖动和重复播放",
      qualityQuestion: "1. 综合动作表现：哪一侧更自然、可信？",
      qualityHelp: "综合考虑姿态、步态、稳定性和明显的重建错误；不要只按轨迹判断。",
      interactionQuestion: "2. 与左侧参考相比，哪一侧更忠实地保留了可见的交互动作？",
      interactionHelp: "判断挥手、观察、起步或避让等可见动作及其发生时机；不是判断动作幅度越大越好。",
      desyncQuestion: "哪一侧的动作—轨迹组合整体更自然？",
      desyncHelp: "直接比较整体协调性即可；如果两侧都自然、都不自然或素材不足，请使用相应选项。",
      aBetter: "A 更好",
      bBetter: "B 更好",
      aMoreNatural: "A 更自然",
      bMoreNatural: "B 更自然",
      similarAcceptable: "一致且都可接受",
      similarPreserved: "一致且都保留",
      bothPoor: "两侧都差",
      bothUnnatural: "两侧都不自然",
      bothFailPreserve: "两侧都未保留",
      noClearInteraction: "参考中无明确交互",
      notEvaluable: "无法评价",
      bothPoorWhy: "两侧都差的原因（至少选一项）",
      bothUnnaturalWhy: "两侧都不自然的原因（至少选一项）",
      preservationFailureWhy: "两侧均未保留的原因（至少选一项）",
      cannotEvaluateWhy: "无法评价的原因",
      chooseOne: "请选择",
      reasonPoseCollapse: "姿态崩坏 / 肢体错位",
      reasonFootSlide: "严重脚底滑动",
      reasonBodyPath: "身体与行进方向失调",
      reasonJitter: "明显抖动 / 跳变",
      reasonOther: "其他方法失败",
      reasonVideoProblem: "视频无法播放或显示异常",
      reasonTooShort: "片段太短",
      reasonOcclusion: "目标遮挡严重",
      reasonTooSmall: "目标太小",
      reasonRepresentation: "当前人体表达不足以判断",
      reasonOtherMaterial: "其他素材原因",
      reasonActionMissing: "动作缺失",
      reasonWrongAction: "动作类型 / 方向错误",
      reasonWrongTiming: "动作时机错误",
      reasonRenderFailure: "重建崩坏遮蔽了动作",
      reasonReferenceUnclear: "参考动作不清晰",
      reasonHeadLimit: "球形头部 / 当前表达无法支持判断",
      reasonGaitPath: "步态与位移不协调",
      reasonStartStop: "起步 / 停步不协调",
      reasonPoseFailure: "人体动作本身崩坏",
      reasonMotionTooSmall: "人物位移 / 动作太小",
      notes: "补充说明",
      optional: "可选",
      notesPlaceholder: "只记录影响判断的重要现象",
      previous: "上一段",
      saveNext: "保存并进入下一段",
      saveFinish: "保存并完成",
      reviewComplete: "评分完成",
      finishCopy: "请下载 CSV 并发回实验组织者。JSON 是完整备份，建议同时保留。",
      finishSummary: "{reviewer} 已完成 {count} / {total} 段评分",
      downloadCsv: "下载 CSV",
      downloadJson: "下载 JSON 备份",
      returnReview: "返回检查评分",
      privacyNote: "导出文件不包含 A/B 的真实方法身份；实验组织者将在所有评分锁定后统一揭盲。",
      sharedSplashTitle: "动作质量与交互保真度",
      sharedSplashCopy: "接下来的每段视频都包含左侧参考和两个匿名结果。A/B 共享同一条根轨迹，因此本部分主要隔离局部人体动作的差异。",
      sharedFocus1: "第一问：哪一侧的整体动作更自然、稳定、可信。",
      sharedFocus2: "第二问：哪一侧更忠实地保留参考中的可见交互动作及其时机。",
      sharedFocus3: "球形头部不表达精细视线方向；无法支持判断时不要猜测。",
      desyncSplashTitle: "动作—轨迹协调性",
      desyncSplashCopy: "接下来请比较两种匿名动作—轨迹组合。只需给出整体自然性判断，无需先说明是否看出了差异，也无需填写判断依据。",
      desyncFocus1: "步态节奏是否与身体位移相符。",
      desyncFocus2: "支撑脚、身体推进、起步与停步是否协调。",
      desyncFocus3: "两侧相近时可选“一致且都可接受”；素材不足时选“无法评价”。",
      beginPartOne: "开始第一部分",
      beginPartTwo: "开始第二部分",
      errorReviewer: "请填写一个匿名评审代号。",
      confirmRestart: "开始新评分会替换此浏览器中尚未导出的进度，确定继续吗？",
      errorQuality: "请选择哪一侧的综合动作表现更好。",
      errorQualityFailure: "选择“两侧都差”时，请至少勾选一个方法失败原因。",
      errorQualityNotEvaluable: "请选择综合动作无法评价的素材原因。",
      errorInteraction: "请选择哪一侧更忠实地保留了可见交互。",
      errorInteractionFailure: "选择“两侧都未保留”时，请至少勾选一个失败原因。",
      errorInteractionNotEvaluable: "请选择交互保真度无法评价的素材原因。",
      errorDesync: "请选择哪一侧的动作—轨迹组合更自然。",
      errorDesyncFailure: "选择“两侧都不自然”时，请至少勾选一个失败原因。",
      errorDesyncNotEvaluable: "请选择动作—轨迹组合无法评价的素材原因。"
    },
    en: {
      pageTitle: "SocialMotion3D · E1c Blind Review",
      brandSubtitle: "E1c blind review",
      notSaved: "Not saved",
      autosaved: "Autosaved",
      exportReminder: "Please export a backup",
      exportCurrent: "Export current results",
      heroTitle: "Does the motion occur naturally<br>along this trajectory?",
      heroCopy: "Estimated time: 25–35 minutes. Results stay in this browser; export the CSV and return it to the study organizer when finished.",
      motionComparison: "motion-method comparisons",
      coherenceCheck: "motion–trajectory checks",
      totalVideos: "videos in total",
      beforeStart: "Before you begin",
      introRule1: "Rate independently, do not discuss responses with other reviewers, and do not search for or open <code>blind_key.json</code>.",
      introRule2: "You may replay each video. A/B are anonymous; when both are comparably acceptable, use “Similar and both acceptable” instead of guessing.",
      introRule3: "The spherical head does not encode gaze direction. Do not judge fine head orientation from it; use “Cannot evaluate” when the material is insufficient.",
      reviewerId: "Reviewer code",
      reviewerPlaceholder: "For example R01 (do not enter your real name)",
      reviewerHelp: "The code is used only to distinguish exported files from different reviewers.",
      startReview: "Start review",
      resumeReview: "Resume {reviewer} ({count} / {total})",
      focusOn: "What to assess",
      partOne: "Part I",
      partTwo: "Part II",
      sampleCount: "Sample {current} / {total}",
      sharedTitle: "Motion comparison on a shared trajectory",
      sharedInstruction: "A and B use exactly the same root trajectory. Judge motion quality separately from the preservation of visible interactions in the reference.",
      sharedLegend: "First assess whether the <strong>overall motion is natural and credible</strong>; then assess whether visible gestures, looking, initiation, or avoidance are <strong>preserved</strong>. Keep the two judgments separate.",
      desyncTitle: "Motion–trajectory coherence comparison",
      desyncInstruction: "Treat A and B as two anonymous motion–trajectory combinations and judge which one is more natural overall.",
      desyncLegend: "You may consider gait versus displacement, support-foot behavior, body progression, and start/stop coordination. You do not need to first report whether a difference is visible.",
      videoFallback: "This browser cannot play the video. Please use a recent version of Edge or Chrome.",
      videoLeft: "Left: reference video",
      videoAB: "Middle / right: anonymous results A / B",
      videoControl: "Pause, seek, and replay as needed",
      qualityQuestion: "1. Overall motion: which side looks more natural and credible?",
      qualityHelp: "Consider pose, gait, stability, and obvious reconstruction failures; do not judge only from the path.",
      interactionQuestion: "2. Compared with the reference on the left, which side better preserves the visible interaction?",
      interactionHelp: "Assess visible gestures, looking, initiation, avoidance, and their timing. Larger motion is not automatically better.",
      desyncQuestion: "Which motion–trajectory combination is more natural overall?",
      desyncHelp: "Make one overall coherence judgment. If both are natural, both unnatural, or the material is insufficient, use the corresponding option.",
      aBetter: "A is better",
      bBetter: "B is better",
      aMoreNatural: "A is more natural",
      bMoreNatural: "B is more natural",
      similarAcceptable: "Similar and both acceptable",
      similarPreserved: "Similar and both preserve it",
      bothPoor: "Both are poor",
      bothUnnatural: "Both are unnatural",
      bothFailPreserve: "Neither preserves it",
      noClearInteraction: "No clear interaction in reference",
      notEvaluable: "Cannot evaluate",
      bothPoorWhy: "Why are both poor? (select at least one)",
      bothUnnaturalWhy: "Why are both unnatural? (select at least one)",
      preservationFailureWhy: "Why does neither preserve it? (select at least one)",
      cannotEvaluateWhy: "Why can this not be evaluated?",
      chooseOne: "Select one",
      reasonPoseCollapse: "Collapsed pose / displaced limbs",
      reasonFootSlide: "Severe foot sliding",
      reasonBodyPath: "Body and travel direction conflict",
      reasonJitter: "Visible jitter / discontinuity",
      reasonOther: "Other method failure",
      reasonVideoProblem: "Video does not play or display correctly",
      reasonTooShort: "Clip is too short",
      reasonOcclusion: "Target is severely occluded",
      reasonTooSmall: "Target is too small",
      reasonRepresentation: "The current body representation is insufficient",
      reasonOtherMaterial: "Other material limitation",
      reasonActionMissing: "Action is missing",
      reasonWrongAction: "Wrong action type / direction",
      reasonWrongTiming: "Wrong action timing",
      reasonRenderFailure: "Reconstruction failure hides the action",
      reasonReferenceUnclear: "Reference action is unclear",
      reasonHeadLimit: "Spherical head / representation cannot support the judgment",
      reasonGaitPath: "Gait and displacement are inconsistent",
      reasonStartStop: "Start / stop behavior is inconsistent",
      reasonPoseFailure: "The body motion itself is broken",
      reasonMotionTooSmall: "Displacement / motion is too small",
      notes: "Additional notes",
      optional: "optional",
      notesPlaceholder: "Record only observations that affected your judgment",
      previous: "Previous",
      saveNext: "Save and continue",
      saveFinish: "Save and finish",
      reviewComplete: "Review complete",
      finishCopy: "Download the CSV and return it to the study organizer. Keep the JSON as a complete backup if possible.",
      finishSummary: "{reviewer} completed {count} / {total} ratings",
      downloadCsv: "Download CSV",
      downloadJson: "Download JSON backup",
      returnReview: "Review last response",
      privacyNote: "The export does not reveal the identities behind A/B. The study organizer will unblind results only after all ratings are locked.",
      sharedSplashTitle: "Motion quality and interaction fidelity",
      sharedSplashCopy: "Each upcoming video contains a reference on the left and two anonymous results. A/B share the same root trajectory, so this section isolates differences in local body motion.",
      sharedFocus1: "Question 1: which side has more natural, stable, and credible overall motion.",
      sharedFocus2: "Question 2: which side better preserves the visible interaction and its timing from the reference.",
      sharedFocus3: "The spherical head does not encode fine gaze direction; do not guess when it cannot support the judgment.",
      desyncSplashTitle: "Motion–trajectory coherence",
      desyncSplashCopy: "Compare two anonymous motion–trajectory combinations. Give one overall naturalness judgment; you do not need to report whether a difference is visible or list your evidence.",
      desyncFocus1: "Whether gait rhythm matches body displacement.",
      desyncFocus2: "Whether support feet, body progression, starts, and stops are coordinated.",
      desyncFocus3: "Use “Similar and both acceptable” when appropriate, or “Cannot evaluate” when the material is insufficient.",
      beginPartOne: "Begin Part I",
      beginPartTwo: "Begin Part II",
      errorReviewer: "Enter an anonymous reviewer code.",
      confirmRestart: "Starting a new review will replace unsent progress stored in this browser. Continue?",
      errorQuality: "Choose which side has better overall motion.",
      errorQualityFailure: "When both are poor, select at least one method-failure reason.",
      errorQualityNotEvaluable: "Select the material limitation that prevents judging overall motion.",
      errorInteraction: "Choose which side better preserves the visible interaction.",
      errorInteractionFailure: "When neither preserves it, select at least one failure reason.",
      errorInteractionNotEvaluable: "Select the material limitation that prevents judging interaction fidelity.",
      errorDesync: "Choose which motion–trajectory combination is more natural.",
      errorDesyncFailure: "When both are unnatural, select at least one failure reason.",
      errorDesyncNotEvaluable: "Select the material limitation that prevents judging motion–trajectory coherence."
    }
  };

  const elements = Object.fromEntries([
    "intro-screen", "section-intro-screen", "review-screen", "finish-screen", "reviewer-id",
    "start-review", "resume-review", "intro-error", "progress-wrap", "progress-text", "progress-bar",
    "autosave-state", "export-top", "section-intro-number", "section-intro-eyebrow", "section-intro-title",
    "section-intro-copy", "section-intro-focus", "begin-section", "section-chip", "sample-count", "task-title",
    "task-instruction", "legend-card", "review-video", "rating-form", "shared-questions", "desync-questions",
    "quality-failure-wrap", "quality-not-evaluable-wrap", "quality-not-evaluable-reason",
    "interaction-failure-wrap", "interaction-not-evaluable-wrap", "interaction-not-evaluable-reason",
    "desync-failure-wrap", "desync-not-evaluable-wrap", "desync-not-evaluable-reason", "notes", "form-error",
    "previous-item", "save-next", "finish-summary", "export-csv", "export-json", "return-review"
  ].map((id) => [id, document.getElementById(id)]));

  let language = localStorage.getItem(LANGUAGE_KEY) === "en" ? "en" : "zh";
  let state = loadState();
  let currentView = "intro";
  let currentSectionIntro = "shared";

  function t(key, values = {}) {
    let value = COPY[language][key] || COPY.zh[key] || key;
    Object.entries(values).forEach(([name, replacement]) => {
      value = value.replaceAll(`{${name}}`, String(replacement));
    });
    return value;
  }

  function applyLanguage() {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.title = t("pageTitle");
    document.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = t(element.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-html]").forEach((element) => {
      element.innerHTML = t(element.dataset.i18nHtml);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
      element.placeholder = t(element.dataset.i18nPlaceholder);
    });
    document.querySelectorAll("[data-language]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.language === language));
    });
    if (state) {
      elements["autosave-state"].textContent = t("autosaved");
      updateProgress();
    }
    refreshIntroDynamic();
    if (currentView === "section") renderSectionIntroContent();
    if (currentView === "review") renderReviewText();
    if (currentView === "finish") renderFinishSummary();
    elements["form-error"].textContent = "";
    elements["intro-error"].textContent = "";
  }

  function createState(reviewerId) {
    return {
      schemaVersion: SCHEMA_VERSION,
      reviewerId,
      sessionId: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`,
      startedAt: new Date().toISOString(),
      completedAt: null,
      currentIndex: 0,
      languageAtStart: language,
      sectionIntroductionsSeen: { shared: false, desync: false },
      answers: {}
    };
  }

  function loadState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (!parsed || parsed.schemaVersion !== SCHEMA_VERSION) return null;
      parsed.sectionIntroductionsSeen ||= { shared: false, desync: false };
      return parsed;
    } catch (_) {
      return null;
    }
  }

  function persistState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      elements["autosave-state"].textContent = t("autosaved");
    } catch (_) {
      elements["autosave-state"].textContent = t("exportReminder");
    }
  }

  function showScreen(name) {
    currentView = name;
    elements["intro-screen"].hidden = name !== "intro";
    elements["section-intro-screen"].hidden = name !== "section";
    elements["review-screen"].hidden = name !== "review";
    elements["finish-screen"].hidden = name !== "finish";
    const active = name !== "intro";
    elements["progress-wrap"].hidden = !active;
    elements["export-top"].hidden = !active;
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function updateProgress() {
    const count = Object.keys(state?.answers || {}).length;
    elements["progress-text"].textContent = `${count} / ${ITEMS.length}`;
    elements["progress-bar"].style.width = `${(count / ITEMS.length) * 100}%`;
  }

  function refreshIntroDynamic() {
    if (!state) return;
    const count = Object.keys(state.answers || {}).length;
    elements["resume-review"].hidden = false;
    elements["resume-review"].textContent = t("resumeReview", {
      reviewer: state.reviewerId,
      count,
      total: ITEMS.length
    });
    elements["reviewer-id"].value = state.reviewerId;
  }

  function start(reviewerId) {
    state = createState(reviewerId.trim());
    persistState();
    updateProgress();
    showSectionIntro("shared");
  }

  function navigateToCurrent() {
    if (!state) return;
    if (state.currentIndex >= ITEMS.length) {
      state.completedAt ||= new Date().toISOString();
      persistState();
      showFinish();
      return;
    }
    if (state.currentIndex === 0 && !state.sectionIntroductionsSeen.shared) {
      showSectionIntro("shared");
      return;
    }
    if (state.currentIndex === CLIPS.length && !state.sectionIntroductionsSeen.desync) {
      showSectionIntro("desync");
      return;
    }
    renderCurrent();
  }

  function showSectionIntro(type) {
    currentSectionIntro = type;
    showScreen("section");
    renderSectionIntroContent();
    updateProgress();
  }

  function renderSectionIntroContent() {
    const shared = currentSectionIntro === "shared";
    elements["section-intro-number"].textContent = shared ? "01" : "02";
    elements["section-intro-eyebrow"].textContent = shared ? "PART I" : "PART II";
    elements["section-intro-title"].textContent = t(shared ? "sharedSplashTitle" : "desyncSplashTitle");
    elements["section-intro-copy"].textContent = t(shared ? "sharedSplashCopy" : "desyncSplashCopy");
    const focusKeys = shared
      ? ["sharedFocus1", "sharedFocus2", "sharedFocus3"]
      : ["desyncFocus1", "desyncFocus2", "desyncFocus3"];
    elements["section-intro-focus"].innerHTML = focusKeys.map((key) => `<li>${t(key)}</li>`).join("");
    elements["begin-section"].textContent = t(shared ? "beginPartOne" : "beginPartTwo");
  }

  function selected(name) {
    return elements["rating-form"].querySelector(`[name="${name}"]:checked`)?.value || "";
  }

  function checkedValues(name) {
    return [...elements["rating-form"].querySelectorAll(`[name="${name}"]:checked`)].map((input) => input.value);
  }

  function setSelected(name, value) {
    elements["rating-form"].querySelectorAll(`[name="${name}"]`).forEach((input) => {
      input.checked = input.value === value;
    });
  }

  function setCheckedValues(name, values) {
    const active = new Set(values || []);
    elements["rating-form"].querySelectorAll(`[name="${name}"]`).forEach((input) => {
      input.checked = active.has(input.value);
    });
  }

  function resetForm() {
    elements["rating-form"].reset();
    [
      "quality-failure-wrap", "quality-not-evaluable-wrap", "interaction-failure-wrap",
      "interaction-not-evaluable-wrap", "desync-failure-wrap", "desync-not-evaluable-wrap"
    ].forEach((id) => { elements[id].hidden = true; });
    elements["form-error"].textContent = "";
  }

  function updateConditionals() {
    const quality = selected("quality_overall");
    const interaction = selected("interaction_fidelity");
    const desync = selected("desync_natural");
    elements["quality-failure-wrap"].hidden = quality !== "both_poor";
    elements["quality-not-evaluable-wrap"].hidden = quality !== "not_evaluable";
    elements["interaction-failure-wrap"].hidden = interaction !== "both_fail";
    elements["interaction-not-evaluable-wrap"].hidden = interaction !== "not_evaluable";
    elements["desync-failure-wrap"].hidden = desync !== "both_poor";
    elements["desync-not-evaluable-wrap"].hidden = desync !== "not_evaluable";
  }

  function videoPath(item) {
    return `videos/${item.type}/${item.file}`;
  }

  function renderReviewText() {
    if (!state || state.currentIndex >= ITEMS.length) return;
    const item = ITEMS[state.currentIndex];
    const shared = item.type === "shared";
    elements["section-chip"].textContent = t(shared ? "partOne" : "partTwo");
    elements["sample-count"].textContent = t("sampleCount", { current: item.sectionIndex + 1, total: CLIPS.length });
    elements["task-title"].textContent = t(shared ? "sharedTitle" : "desyncTitle");
    elements["task-instruction"].textContent = t(shared ? "sharedInstruction" : "desyncInstruction");
    elements["legend-card"].innerHTML = t(shared ? "sharedLegend" : "desyncLegend");
    elements["save-next"].textContent = t(state.currentIndex === ITEMS.length - 1 ? "saveFinish" : "saveNext");
  }

  function renderCurrent() {
    if (!state) return;
    if (state.currentIndex >= ITEMS.length) {
      showFinish();
      return;
    }
    const item = ITEMS[state.currentIndex];
    showScreen("review");
    resetForm();
    elements["shared-questions"].hidden = item.type !== "shared";
    elements["desync-questions"].hidden = item.type !== "desync";
    elements["review-video"].src = videoPath(item);
    elements["review-video"].load();
    elements["previous-item"].disabled = state.currentIndex === 0;
    restoreAnswer(item);
    renderReviewText();
    updateProgress();
  }

  function restoreAnswer(item) {
    const answer = state.answers[item.id];
    if (!answer) return;
    if (item.type === "shared") {
      setSelected("quality_overall", answer.qualityOverall || "");
      setCheckedValues("quality_failure_reason", answer.qualityFailureReasons);
      elements["quality-not-evaluable-reason"].value = answer.qualityNotEvaluableReason || "";
      setSelected("interaction_fidelity", answer.interactionFidelity || "");
      setCheckedValues("interaction_failure_reason", answer.interactionFailureReasons);
      elements["interaction-not-evaluable-reason"].value = answer.interactionNotEvaluableReason || "";
    } else {
      setSelected("desync_natural", answer.desyncNatural || "");
      setCheckedValues("desync_failure_reason", answer.desyncFailureReasons);
      elements["desync-not-evaluable-reason"].value = answer.desyncNotEvaluableReason || "";
    }
    elements.notes.value = answer.notes || "";
    updateConditionals();
  }

  function validate(item) {
    if (item.type === "shared") {
      const quality = selected("quality_overall");
      if (!quality) return t("errorQuality");
      if (quality === "both_poor" && !checkedValues("quality_failure_reason").length) return t("errorQualityFailure");
      if (quality === "not_evaluable" && !elements["quality-not-evaluable-reason"].value) return t("errorQualityNotEvaluable");
      const interaction = selected("interaction_fidelity");
      if (!interaction) return t("errorInteraction");
      if (interaction === "both_fail" && !checkedValues("interaction_failure_reason").length) return t("errorInteractionFailure");
      if (interaction === "not_evaluable" && !elements["interaction-not-evaluable-reason"].value) return t("errorInteractionNotEvaluable");
      return "";
    }
    const desync = selected("desync_natural");
    if (!desync) return t("errorDesync");
    if (desync === "both_poor" && !checkedValues("desync_failure_reason").length) return t("errorDesyncFailure");
    if (desync === "not_evaluable" && !elements["desync-not-evaluable-reason"].value) return t("errorDesyncNotEvaluable");
    return "";
  }

  function collectAnswer(item) {
    const base = {
      itemId: item.id,
      taskType: item.type,
      sectionIndex: item.sectionIndex + 1,
      videoFile: item.file,
      interfaceLanguage: language,
      notes: elements.notes.value.trim(),
      ratedAt: new Date().toISOString()
    };
    if (item.type === "shared") {
      const quality = selected("quality_overall");
      const interaction = selected("interaction_fidelity");
      return {
        ...base,
        qualityOverall: quality,
        qualityFailureReasons: quality === "both_poor" ? checkedValues("quality_failure_reason") : [],
        qualityNotEvaluableReason: quality === "not_evaluable" ? elements["quality-not-evaluable-reason"].value : "",
        interactionFidelity: interaction,
        interactionFailureReasons: interaction === "both_fail" ? checkedValues("interaction_failure_reason") : [],
        interactionNotEvaluableReason: interaction === "not_evaluable" ? elements["interaction-not-evaluable-reason"].value : ""
      };
    }
    const desync = selected("desync_natural");
    return {
      ...base,
      desyncNatural: desync,
      desyncFailureReasons: desync === "both_poor" ? checkedValues("desync_failure_reason") : [],
      desyncNotEvaluableReason: desync === "not_evaluable" ? elements["desync-not-evaluable-reason"].value : ""
    };
  }

  function answerRows() {
    return ITEMS.filter((item) => state.answers[item.id]).map((item) => {
      const answer = state.answers[item.id];
      return {
        schema_version: SCHEMA_VERSION,
        reviewer_id: state.reviewerId,
        session_id: state.sessionId,
        task_type: item.type === "shared" ? "gem_vs_motionbert_shared_trajectory" : "motion_trajectory_coherence",
        sample_number: item.sectionIndex + 1,
        video_file: item.file,
        overall_motion_choice: answer.qualityOverall || "",
        overall_motion_failure_reasons: (answer.qualityFailureReasons || []).join("|"),
        overall_motion_not_evaluable_reason: answer.qualityNotEvaluableReason || "",
        interaction_fidelity_choice: answer.interactionFidelity || "",
        interaction_failure_reasons: (answer.interactionFailureReasons || []).join("|"),
        interaction_not_evaluable_reason: answer.interactionNotEvaluableReason || "",
        motion_trajectory_naturalness_choice: answer.desyncNatural || "",
        motion_trajectory_failure_reasons: (answer.desyncFailureReasons || []).join("|"),
        motion_trajectory_not_evaluable_reason: answer.desyncNotEvaluableReason || "",
        interface_language: answer.interfaceLanguage || "",
        notes: answer.notes || "",
        rated_at: answer.ratedAt
      };
    });
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }

  function safeReviewer() {
    return (state.reviewerId || "reviewer").replace(/[^a-zA-Z0-9_-]+/g, "_").slice(0, 40);
  }

  function download(content, mime, suffix) {
    const complete = Object.keys(state.answers).length === ITEMS.length ? "complete" : "partial";
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `SocialMotion3D_E1c_${safeReviewer()}_${complete}.${suffix}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function exportCsv() {
    const rows = answerRows();
    if (!rows.length) return;
    const headers = Object.keys(rows[0]);
    const csv = [headers.join(","), ...rows.map((row) => headers.map((key) => csvEscape(row[key])).join(","))].join("\r\n");
    download(`\ufeff${csv}`, "text/csv;charset=utf-8", "csv");
  }

  function exportJson() {
    if (!state) return;
    const payload = { ...state, exportedAt: new Date().toISOString(), itemCount: ITEMS.length };
    download(JSON.stringify(payload, null, 2), "application/json;charset=utf-8", "json");
  }

  function renderFinishSummary() {
    elements["finish-summary"].textContent = t("finishSummary", {
      reviewer: state.reviewerId,
      count: Object.keys(state.answers).length,
      total: ITEMS.length
    });
  }

  function showFinish() {
    updateProgress();
    showScreen("finish");
    renderFinishSummary();
  }

  elements["start-review"].addEventListener("click", () => {
    const reviewerId = elements["reviewer-id"].value.trim();
    if (!reviewerId) {
      elements["intro-error"].textContent = t("errorReviewer");
      elements["reviewer-id"].focus();
      return;
    }
    if (state && Object.keys(state.answers || {}).length && !window.confirm(t("confirmRestart"))) return;
    start(reviewerId);
  });

  elements["resume-review"].addEventListener("click", navigateToCurrent);

  elements["begin-section"].addEventListener("click", () => {
    state.sectionIntroductionsSeen[currentSectionIntro] = true;
    persistState();
    renderCurrent();
  });

  elements["rating-form"].addEventListener("change", (event) => {
    if (["quality_overall", "interaction_fidelity", "desync_natural"].includes(event.target.name)) {
      updateConditionals();
    }
  });

  elements["rating-form"].addEventListener("submit", (event) => {
    event.preventDefault();
    const item = ITEMS[state.currentIndex];
    const error = validate(item);
    elements["form-error"].textContent = error;
    if (error) return;
    state.answers[item.id] = collectAnswer(item);
    state.currentIndex += 1;
    if (state.currentIndex >= ITEMS.length) state.completedAt = new Date().toISOString();
    persistState();
    navigateToCurrent();
  });

  elements["previous-item"].addEventListener("click", () => {
    if (state.currentIndex > 0) {
      state.currentIndex -= 1;
      persistState();
      renderCurrent();
    }
  });

  elements["export-top"].addEventListener("click", exportCsv);
  elements["export-csv"].addEventListener("click", exportCsv);
  elements["export-json"].addEventListener("click", exportJson);
  elements["return-review"].addEventListener("click", () => {
    state.currentIndex = ITEMS.length - 1;
    persistState();
    renderCurrent();
  });

  document.querySelectorAll("[data-language]").forEach((button) => {
    button.addEventListener("click", () => {
      language = button.dataset.language;
      localStorage.setItem(LANGUAGE_KEY, language);
      applyLanguage();
    });
  });

  applyLanguage();
})();
