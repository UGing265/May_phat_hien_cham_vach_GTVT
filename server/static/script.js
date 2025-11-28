// server/static/script.js

async function callApi(path, options = {}) {
  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    return await res.json();
  } catch (err) {
    console.error(err);
    showMessage("Không gọi được API. Kiểm tra mạng / server.");
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
  const btnUsePhoneCam = document.getElementById("btn-use-phone-cam");

  const phoneCamContainer = document.getElementById("phone-cam-container");
  const phoneVideo = document.getElementById("phone-video");
  const phoneCanvas = document.getElementById("phone-canvas");

  let phoneStream = null;
  let phoneSendTimer = null;

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
    const res = await fetch("/api/play-sound", { method: "POST" });
    let data = null;
    try {
      data = await res.json();
    } catch {}
    if (data && data.message) {
      showMessage(data.message);
    } else {
      showMessage("Đã gửi lệnh phát âm thanh.");
    }
  });

  async function startPhoneCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showMessage("Trình duyệt không hỗ trợ camera (getUserMedia).");
      return;
    }

    try {
      phoneStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }, // camera sau
        audio: false,
      });
      phoneVideo.srcObject = phoneStream;
      phoneCamContainer.style.display = "block";
      showMessage("Đã bật camera iPhone. Đang gửi frame về server...");

      startSendingFramesFromPhone();
    } catch (err) {
      console.error(err);
      showMessage(
        "Không bật được camera iPhone. Kiểm tra quyền truy cập camera / HTTPS."
      );
    }
  }

  function startSendingFramesFromPhone() {
    if (phoneSendTimer) clearInterval(phoneSendTimer);

    phoneSendTimer = setInterval(() => {
      if (!phoneVideo.videoWidth || !phoneVideo.videoHeight) return;

      const w = phoneVideo.videoWidth;
      const h = phoneVideo.videoHeight;

      phoneCanvas.width = w;
      phoneCanvas.height = h;
      const ctx = phoneCanvas.getContext("2d");
      ctx.drawImage(phoneVideo, 0, 0, w, h);

      phoneCanvas.toBlob(
        (blob) => {
          if (!blob) return;
          const formData = new FormData();
          formData.append("frame", blob, "frame.jpg");

          fetch("/api/upload-frame", {
            method: "POST",
            body: formData,
          }).catch((e) => console.error(e));
        },
        "image/jpeg",
        0.6
      );
    }, 500); // 2 fps cho nhẹ. Sau muốn mượt thì giảm xuống 200-300ms
  }

  btnUsePhoneCam.addEventListener("click", () => {
    if (!phoneStream) {
      startPhoneCamera();
    } else {
      showMessage("Camera iPhone đã bật rồi.");
    }
  });

  // Load trạng thái ban đầu
  (async () => {
    const data = await callApi("/api/status");
    if (data) updateStatusUI(data.running);
  })();
});
