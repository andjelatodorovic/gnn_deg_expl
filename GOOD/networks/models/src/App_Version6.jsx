// Replace the existing submitForm function in your file with the one below

async function submitForm(e) {
  e.preventDefault();
  if (!affiliation || affiliation.trim() === "") {
    // basic client validation
    alert("Please provide your institutional affiliation.");
    return;
  }

  setSubmitted(true);

  try {
    const resp = await fetch("/api/express-interest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ affiliation: affiliation.trim(), email: (email || "").trim() })
    });

    const data = await resp.json().catch(() => ({}));
    if (resp.ok) {
      // optional: show a brief success message before closing
      // you can replace this with a nicer UI/alert component
      // keep modal open briefly so user sees confirmation
      setTimeout(() => {
        setOpen(false);
        setSubmitted(false);
        setAffiliation("");
        setEmail("");
      }, 900);
    } else {
      console.error("Submission error:", data);
      alert(data.error || "Submission failed. Please try again.");
      setSubmitted(false);
    }
  } catch (err) {
    console.error("Network error:", err);
    alert("Network error. Please try again.");
    setSubmitted(false);
  }
}