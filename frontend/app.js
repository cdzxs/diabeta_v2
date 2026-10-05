(function () {
    const storageKey = "diabetaRecords";

    function readRecords() {
        try {
            const records = JSON.parse(localStorage.getItem(storageKey));
            return Array.isArray(records) ? records : [];
        } catch (error) {
            return [];
        }
    }

    function writeRecords(records) {
        localStorage.setItem(storageKey, JSON.stringify(records));
    }

    function formatDate(value) {
        return new Intl.DateTimeFormat(undefined, {
            dateStyle: "medium",
            timeStyle: "short"
        }).format(new Date(value));
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function personRows(person) {
        // Stage 1 is the five-input screener only; never surface lab placeholders
        return Object.entries(person || {}).filter(
            ([label, value]) =>
                !(label === "HbA1c" && value === "Not provided") &&
                !(label === "Fasting Glucose" && value === "Not provided")
        );
    }

    function buildReportHtml(record) {
        const detailRows = personRows(record.person)
            .map(([label, value]) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`)
            .join("");

        const reasons = Array.isArray(record.reason) && record.reason.length
            ? record.reason.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
            : "<li>Model-driven result based on submitted values.</li>";

        const factors = Array.isArray(record.shap) && record.shap.length
            ? record.shap.map((item) => `<li>${escapeHtml(item.feature)}: ${escapeHtml(item.direction)} contribution</li>`).join("")
            : "<li>No model factor breakdown was generated for this rule-based result.</li>";

        return `<!DOCTYPE html>
<html>
<head>
    <title>DiaBeta Result</title>
    <style>
        body { color: #14213d; font-family: Arial, sans-serif; margin: 40px; }
        .brand { color: #0e2f56; font-size: 28px; font-weight: 800; }
        .meta { color: #607084; margin: 8px 0 28px; }
        .result { border: 1px solid #dce6ec; border-left: 8px solid #1767b3; padding: 20px; margin-bottom: 24px; }
        .result strong { display: block; font-size: 34px; margin-bottom: 6px; }
        table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; }
        th, td { border: 1px solid #dce6ec; padding: 10px; text-align: left; }
        th { width: 42%; background: #f5faf9; }
        h2 { margin-top: 28px; }
        li { margin: 8px 0; }
        .disclaimer { color: #607084; font-size: 13px; margin-top: 36px; }
    </style>
</head>
<body>
    <div class="brand">DiaBeta Diabetes Risk Assessment</div>
    <div class="meta">Generated ${escapeHtml(formatDate(record.createdAt))}</div>
    <section class="result">
        <strong>${escapeHtml(record.tier)} Risk</strong>
        <span>${escapeHtml(record.probability)}% high-risk probability</span>
    </section>
    <h2>Submitted Details</h2>
    <table>${detailRows}</table>
    <h2>Clinical Notes</h2>
    <ul>${reasons}</ul>
    <h2>Top Factors</h2>
    <ul>${factors}</ul>
    <p class="disclaimer">DiaBeta is a research and education tool. It does not diagnose medical conditions and does not replace professional medical advice, diagnosis, or lab tests.</p>
</body>
</html>`;
    }

    function downloadRecord(record) {
        const reportWindow = window.open("", "_blank");
        if (!reportWindow) return;
        reportWindow.document.open();
        reportWindow.document.write(buildReportHtml(record));
        reportWindow.document.close();
        reportWindow.focus();
        setTimeout(() => reportWindow.print(), 250);
    }

    function setupNavDrawer() {
        const toggle = document.querySelector('[data-drawer-toggle]');
        const drawer = document.querySelector('.side-drawer');
        const overlay = document.querySelector('.drawer-overlay');
        const closeBtn = document.querySelector('[data-drawer-close]');
        if (!toggle || !drawer || !overlay) return;

        const closeDrawer = () => {
            drawer.classList.remove('open');
            overlay.classList.remove('open');
            drawer.setAttribute('aria-hidden', 'true');
            toggle.setAttribute('aria-expanded', 'false');
            document.body.classList.remove('drawer-open');
        };

        const openDrawer = () => {
            drawer.classList.add('open');
            overlay.classList.add('open');
            drawer.setAttribute('aria-hidden', 'false');
            toggle.setAttribute('aria-expanded', 'true');
            document.body.classList.add('drawer-open');
        };

        toggle.addEventListener('click', openDrawer);
        closeBtn?.addEventListener('click', closeDrawer);
        overlay.addEventListener('click', closeDrawer);
        drawer.querySelectorAll('a, button').forEach((link) => link.addEventListener('click', closeDrawer));
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') closeDrawer();
        });
    }

    function buildRecordPreview(record, index) {
        const detailRows = personRows(record.person)
            .map(([label, value]) => `
                <div>
                    <dt>${escapeHtml(label)}</dt>
                    <dd>${escapeHtml(value)}</dd>
                </div>
            `).join("");

        const notes = Array.isArray(record.reason) && record.reason.length
            ? record.reason.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
            : "<li>Model-driven result based on submitted values.</li>";

        return `
            <div class="record-preview">
                <p class="record-date">${escapeHtml(formatDate(record.createdAt))}</p>
                <h2 id="recordModalTitle">${escapeHtml(record.tier)} Risk</h2>
                <p class="preview-score">${escapeHtml(record.probability)}% high-risk probability</p>
                <dl class="input-list">${detailRows}</dl>
                <div class="preview-notes">
                    <h3>Clinical Notes</h3>
                    <ul>${notes}</ul>
                </div>
                <button class="primary-action" type="button" data-modal-download="${index}">Download PDF</button>
            </div>
        `;
    }

    function openRecordModal(record, index) {
        const modal = document.getElementById("recordModal");
        const content = document.getElementById("recordModalContent");
        if (!modal || !content) return;

        content.innerHTML = buildRecordPreview(record, index);
        modal.hidden = false;
        document.body.classList.add("modal-open");
        content.querySelector("[data-modal-download]")?.addEventListener("click", () => downloadRecord(record));
    }

    function closeRecordModal() {
        const modal = document.getElementById("recordModal");
        if (!modal) return;
        modal.hidden = true;
        document.body.classList.remove("modal-open");
    }

    function renderRecords() {
        const list = document.getElementById("recordsList");
        if (!list) return;

        const records = readRecords();
        if (!records.length) {
            list.innerHTML = `
                <div class="empty-state">
                    <h2>No records yet</h2>
                    <p>Run an assessment and the result will appear here automatically.</p>
                    <a class="primary-link" href="stage1.html">Start Assessment</a>
                </div>`;
            return;
        }

        list.innerHTML = records.map((record, index) => `
            <article class="record-card">
                <div>
                    <p class="record-date">${escapeHtml(formatDate(record.createdAt))}</p>
                    <h2>${escapeHtml(record.tier)} Risk</h2>
                    <p>${escapeHtml(record.probability)}% high-risk probability</p>
                </div>
                <div class="record-actions">
                    <button class="secondary-action" type="button" data-view-record="${index}">View</button>
                    <button class="secondary-action" type="button" data-record-index="${index}">Download PDF</button>
                    <button class="secondary-action danger-action" type="button" data-delete-record="${index}">Delete</button>
                </div>
            </article>
        `).join("");

        list.querySelectorAll("[data-view-record]").forEach((button) => {
            button.addEventListener("click", () => {
                const index = Number(button.dataset.viewRecord);
                const record = readRecords()[index];
                if (record) openRecordModal(record, index);
            });
        });

        list.querySelectorAll("[data-record-index]").forEach((button) => {
            button.addEventListener("click", () => {
                const record = readRecords()[Number(button.dataset.recordIndex)];
                if (record) downloadRecord(record);
            });
        });

        list.querySelectorAll("[data-delete-record]").forEach((button) => {
            button.addEventListener("click", () => {
                const index = Number(button.dataset.deleteRecord);
                if (!window.confirm("Delete this record?")) return;
                const records = readRecords();
                if (index < 0 || index >= records.length) return;
                records.splice(index, 1);
                writeRecords(records);
                renderRecords();
            });
        });
    }

    // Records are saved exactly once, by the Stage 1 submit handler
    // (stage1.html) at assessment time. result.html only READS
    // diabeta_last_result to display the outcome - it must not write a
    // second copy into diabetaRecords.
    renderRecords();

    window.addEventListener('storage', event => {
        if (!event.key || event.key === storageKey) {
            closeRecordModal();
            renderRecords();
        }
    });

    document.querySelector("[data-download-current]")?.addEventListener("click", () => {
        if (window.DIABETA_RESULT) downloadRecord(window.DIABETA_RESULT);
    });

    document.querySelectorAll("[data-clear-records]").forEach((button) => {
        button.addEventListener("click", () => {
            if (!window.confirm("Clear all saved assessment records? This cannot be undone.")) return;
            localStorage.removeItem(storageKey);
            closeRecordModal();
            renderRecords();
        });
    });

    document.querySelector("[data-close-record]")?.addEventListener("click", closeRecordModal);
    document.getElementById("recordModal")?.addEventListener("click", (event) => {
        if (event.target.id === "recordModal") closeRecordModal();
    });

    setupNavDrawer();
})();
