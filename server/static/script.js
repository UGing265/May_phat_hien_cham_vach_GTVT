async function callApi(path, options = {}) {
  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    return await res.json();
  } catch (err) {
    console.error(err);
    showMessage("Không gọi được API. Kiểm tra mạng / Raspberry Pi.");
    return null;
  }
}

function updateStatusUI(running) {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");

  if (running) {
    dot.classList.add("running");
    text.textContent = "Đang chạy";
  } else {
    dot.classList.remove("running");
    text.textContent = "Đang dừng";
  }
}

function showMessage(msg) {
  const el = document.getElementById("message");
  el.textContent = msg || "";
}

document.addEventListener("DOMContentLoaded", () => {
  const btnStart = document.getElementById("btn-start");
  const btnStop = document.getElementById("btn-stop");
  const btnSound = document.getElementById("btn-sound");

  btnStart.addEventListener("click", async () => {
    const data = await callApi("/api/start", { method: "POST" });
    if (data) {
      updateStatusUI(data.running);
      showMessage("Đã gửi lệnh START.");
    }
  });

  btnStop.addEventListener("click", async () => {
    const data = await callApi("/api/stop", { method: "POST" });
    if (data) {
      updateStatusUI(data.running);
      showMessage("Đã gửi lệnh STOP.");
    }
  });

  btnSound.addEventListener("click", async () => {
    const data = await callApi("/api/play-sound", { method: "POST" });
    if (data) {
      showMessage(data.message || "Đã phát thử âm thanh.");
    }
  });

  // load trạng thái ban đầu
  (async () => {
    const data = await callApi("/api/status");
    if (data) updateStatusUI(data.running);
  })();

  // poll status mỗi 3s (nếu sau này detector tự tắt)
  setInterval(async () => {
    const data = await callApi("/api/status");
    if (data) updateStatusUI(data.running);
  }, 3000);
});
