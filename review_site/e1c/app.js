(function () {
  "use strict";

  const SCHEMA_VERSION = "e1c-human-review-1.0";
  const STORAGE_KEY = "socialmotion3d-e1c-human-review-v1";
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
    "23_gp_set_0005_vid_0005_gp_5751_OCCLUSION_NIGHT_night_high_blind.mp4"
  ];

  const ITEMS = [
    ...CLIPS.map((file, index) => ({ id: `shared-${index + 1}`, type: "shared", file, sectionIndex: index })),
    ...CLIPS.map((file, index) => ({ id: `desync-${index + 1}`, type: "desync", file, sectionIndex: index }))
  ];

  const elements = Object.fromEntries([
    "intro-screen", "review-screen", "finish-screen", "reviewer-id", "start-review", "resume-review",
    "intro-error", "progress-wrap", "progress-text", "progress-bar", "autosave-state", "export-top",
    "section-chip", "sample-count", "task-title", "task-instruction", "legend-card", "review-video",
    "rating-form", "not-evaluable-wrap", "not-evaluable-reason", "shared-questions", "desync-questions",
    "notes", "form-error", "previous-item", "save-next", "finish-summary", "export-csv", "export-json",
    "return-review"
  ].map((id) => [id, document.getElementById(id)]));

  let state = loadState();

  function createState(reviewerId) {
    return {
      schemaVersion: SCHEMA_VERSION,
      reviewerId,
      sessionId: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`,
      startedAt: new Date().toISOString(),
      completedAt: null,
      currentIndex: 0,
      answers: {}
    };
  }

  function loadState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY));
      return parsed && parsed.schemaVersion === SCHEMA_VERSION ? parsed : null;
    } catch (_) {
      return null;
    }
  }

  function persistState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      elements["autosave-state"].textContent = "已自动保存";
    } catch (_) {
      elements["autosave-state"].textContent = "请及时导出备份";
    }
  }

  function showScreen(name) {
    elements["intro-screen"].hidden = name !== "intro";
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

  function start(reviewerId) {
    state = createState(reviewerId.trim());
    persistState();
    showScreen("review");
    renderCurrent();
  }

  function selected(name) {
    return elements["rating-form"].querySelector(`[name="${name}"]:checked`)?.value || "";
  }

  function setSelected(name, value) {
    elements["rating-form"].querySelectorAll(`[name="${name}"]`).forEach((input) => {
      input.checked = input.value === value;
    });
  }

  function setFormEnabled(enabled) {
    elements["shared-questions"].querySelectorAll("input, select").forEach((input) => { input.disabled = !enabled; });
    elements["desync-questions"].querySelectorAll("input, select").forEach((input) => { input.disabled = !enabled; });
  }

  function resetForm() {
    elements["rating-form"].reset();
    elements["not-evaluable-wrap"].hidden = true;
    elements["form-error"].textContent = "";
    setFormEnabled(true);
  }

  function videoPath(item) {
    return `videos/${item.type}/${item.file}`;
  }

  function renderCurrent() {
    if (!state) return;
    if (state.currentIndex >= ITEMS.length) {
      state.completedAt ||= new Date().toISOString();
      persistState();
      showFinish();
      return;
    }
    const item = ITEMS[state.currentIndex];
    showScreen("review");
    resetForm();
    elements["section-chip"].textContent = item.type === "shared" ? "第一部分" : "第二部分";
    elements["sample-count"].textContent = `样本 ${item.sectionIndex + 1} / ${CLIPS.length}`;
    elements["task-title"].textContent = item.type === "shared" ? "共享轨迹上的动作比较" : "动作—轨迹同步敏感性检查";
    elements["task-instruction"].textContent = item.type === "shared"
      ? "A/B 使用完全相同的根轨迹。请比较局部人体动作与参考视频、脚步和行进方向的协调程度。"
      : "A/B 来自同一动作与同一根轨迹，但其中一侧的局部动作相位被人为错开。请判断哪一侧与轨迹配合得更自然。";
    elements["legend-card"].innerHTML = item.type === "shared"
      ? "重点观察：<strong>脚底是否滑动</strong>、步态是否推动身体前进、身体朝向与轨迹是否持续失调、起步与停步是否符合参考。"
      : "这不是轨迹速度修改。人为处理只破坏<strong>局部动作与原轨迹的时间同步</strong>；如果确实看不出来，请选择“无法分辨”。";
    elements["shared-questions"].hidden = item.type !== "shared";
    elements["desync-questions"].hidden = item.type !== "desync";
    elements["review-video"].src = videoPath(item);
    elements["review-video"].load();
    elements["previous-item"].disabled = state.currentIndex === 0;
    elements["save-next"].textContent = state.currentIndex === ITEMS.length - 1 ? "保存并完成" : "保存并进入下一段";
    restoreAnswer(item);
    updateProgress();
  }

  function restoreAnswer(item) {
    const answer = state.answers[item.id];
    if (!answer) return;
    setSelected("evaluable", answer.evaluable);
    elements["not-evaluable-reason"].value = answer.notEvaluableReason || "";
    elements["not-evaluable-wrap"].hidden = answer.evaluable !== "no";
    setFormEnabled(answer.evaluable !== "no");
    if (item.type === "shared") {
      ["foot_slide", "body_path_mismatch", "reference_timing_mismatch"].forEach((name) => {
        elements["rating-form"].elements[name].value = answer[name] || "";
      });
      setSelected("shared_overall", answer.sharedOverall || "");
    } else {
      setSelected("difference_visible", answer.differenceVisible || "");
      setSelected("desync_natural", answer.desyncNatural || "");
      const evidence = new Set(answer.desyncEvidence || []);
      elements["rating-form"].querySelectorAll('[name="desync_evidence"]').forEach((input) => {
        input.checked = evidence.has(input.value);
      });
    }
    elements.notes.value = answer.notes || "";
  }

  function validate(item) {
    const evaluable = selected("evaluable");
    if (!evaluable) return "请先判断这段视频能否正常评价。";
    if (evaluable === "no") {
      return elements["not-evaluable-reason"].value ? "" : "请选择无法评价的主要原因。";
    }
    if (item.type === "shared") {
      const missingIssue = [...elements["rating-form"].querySelectorAll("[data-shared-required]")].some((select) => !select.value);
      if (missingIssue) return "请完成三项明显问题检查。";
      if (!selected("shared_overall")) return "请选择综合表现，无法判断时请选择“无明显差异”。";
    } else {
      if (!selected("difference_visible")) return "请选择是否能够看出同步差异。";
      if (!selected("desync_natural")) return "请选择更自然的一侧，无法判断时请选择“无法分辨”。";
    }
    return "";
  }

  function collectAnswer(item) {
    const evaluable = selected("evaluable");
    const base = {
      itemId: item.id,
      taskType: item.type,
      sectionIndex: item.sectionIndex + 1,
      videoFile: item.file,
      evaluable,
      notEvaluableReason: evaluable === "no" ? elements["not-evaluable-reason"].value : "",
      notes: elements.notes.value.trim(),
      ratedAt: new Date().toISOString()
    };
    if (item.type === "shared") {
      return {
        ...base,
        foot_slide: elements["rating-form"].elements.foot_slide.value,
        body_path_mismatch: elements["rating-form"].elements.body_path_mismatch.value,
        reference_timing_mismatch: elements["rating-form"].elements.reference_timing_mismatch.value,
        sharedOverall: selected("shared_overall")
      };
    }
    return {
      ...base,
      differenceVisible: selected("difference_visible"),
      desyncNatural: selected("desync_natural"),
      desyncEvidence: [...elements["rating-form"].querySelectorAll('[name="desync_evidence"]:checked')].map((input) => input.value)
    };
  }

  function issueFlags(value) {
    return {
      a: value === "a" || value === "both",
      b: value === "b" || value === "both"
    };
  }

  function answerRows() {
    return ITEMS.filter((item) => state.answers[item.id]).map((item) => {
      const answer = state.answers[item.id];
      const foot = issueFlags(answer.foot_slide);
      const path = issueFlags(answer.body_path_mismatch);
      const timing = issueFlags(answer.reference_timing_mismatch);
      return {
        schema_version: SCHEMA_VERSION,
        reviewer_id: state.reviewerId,
        session_id: state.sessionId,
        task_type: item.type === "shared" ? "gem_vs_motionbert_shared_trajectory" : "native_vs_desynchronized",
        sample_number: item.sectionIndex + 1,
        video_file: item.file,
        evaluable: answer.evaluable,
        not_evaluable_reason: answer.notEvaluableReason || "",
        foot_slide_a: answer.taskType === "shared" ? foot.a : "",
        foot_slide_b: answer.taskType === "shared" ? foot.b : "",
        body_path_mismatch_a: answer.taskType === "shared" ? path.a : "",
        body_path_mismatch_b: answer.taskType === "shared" ? path.b : "",
        reference_timing_mismatch_a: answer.taskType === "shared" ? timing.a : "",
        reference_timing_mismatch_b: answer.taskType === "shared" ? timing.b : "",
        shared_overall_choice: answer.sharedOverall || "",
        synchronization_difference_visible: answer.differenceVisible || "",
        desync_more_natural_choice: answer.desyncNatural || "",
        desync_evidence: (answer.desyncEvidence || []).join("|"),
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

  function showFinish() {
    updateProgress();
    showScreen("finish");
    elements["finish-summary"].textContent = `${state.reviewerId} 已完成 ${Object.keys(state.answers).length} / ${ITEMS.length} 段评分`;
  }

  elements["start-review"].addEventListener("click", () => {
    const reviewerId = elements["reviewer-id"].value.trim();
    if (!reviewerId) {
      elements["intro-error"].textContent = "请填写一个匿名评审代号。";
      elements["reviewer-id"].focus();
      return;
    }
    if (state && Object.keys(state.answers || {}).length && !window.confirm("开始新评分会替换此浏览器中尚未导出的进度，确定继续吗？")) return;
    start(reviewerId);
  });

  elements["resume-review"].addEventListener("click", () => {
    showScreen(state.currentIndex >= ITEMS.length ? "finish" : "review");
    state.currentIndex >= ITEMS.length ? showFinish() : renderCurrent();
  });

  elements["rating-form"].addEventListener("change", (event) => {
    if (event.target.name !== "evaluable") return;
    const evaluable = selected("evaluable");
    elements["not-evaluable-wrap"].hidden = evaluable !== "no";
    setFormEnabled(evaluable !== "no");
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
    renderCurrent();
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

  if (state) {
    const count = Object.keys(state.answers || {}).length;
    elements["resume-review"].hidden = false;
    elements["resume-review"].textContent = `继续 ${state.reviewerId} 的评分（${count} / ${ITEMS.length}）`;
    elements["reviewer-id"].value = state.reviewerId;
  }
})();
