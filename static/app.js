let currentVideoId = null;

async function uploadVideo() {
  const fileInput = document.getElementById("videoInput");
  const status = document.getElementById("uploadStatus");

  if (!fileInput.files.length) {
    status.textContent = "Please select a video file.";
    return;
  }

  const formData = new FormData();
  formData.append("video", fileInput.files[0]);

  status.textContent = "Uploading and processing...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      currentVideoId = null;
      status.textContent = data.error || "Upload failed. Please try again.";
      return;
    }

    currentVideoId = data.video_id;
    status.textContent = "Video processed. You can now ask questions.";
  } catch (err) {
    currentVideoId = null;
    status.textContent = "Network error during upload. Please try again.";
  }
}

async function askQuestion() {
  const question = document.getElementById("questionInput").value;
  const answerBox = document.getElementById("answerBox");

  if (!currentVideoId) {
    answerBox.textContent = "Please upload a video first.";
    return;
  }

  answerBox.textContent = "Loading...";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        video_id: currentVideoId,
      }),
    });

    let data;
    try {
      data = await response.json();
    } catch (_err) {
      answerBox.textContent = "Server returned non-JSON response.";
      return;
    }

    if (!response.ok) {
      answerBox.textContent = data.error || "Question failed. Please try again.";
      // clear id if server says video missing
      if (data.error === "Video not processed") currentVideoId = null;
      return;
    }

    answerBox.textContent = data.answer || "(no answer returned)";
  } catch (err) {
    answerBox.textContent = "Network error while asking. Please try again.";
  }
}
