import os
import json
import time
import chainlit as cl
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# Gemini Setup
# ============================================================

load_dotenv(".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.8-flash"

HISTORY_FILE = "patient_history.json"
CUSTOM_PATIENTS_FILE = "custom_patients.json"


# ============================================================
# Gemini Retry Functions
# ============================================================

def is_temporary_gemini_error(error):
    """
    Determine whether a Gemini error appears to be temporary.
    """

    error_text = str(error).lower()

    return (
        "503" in error_text
        or "429" in error_text
        or "unavailable" in error_text
        or "high demand" in error_text
        or "resource_exhausted" in error_text
        or "quota exceeded" in error_text
        or "temporarily" in error_text
    )


def send_with_retry(chat, message, max_attempts=3):
    """
    Send a message through a Gemini chat session.

    Temporary 503 errors are retried automatically.
    """

    for attempt in range(1, max_attempts + 1):

        try:
            return chat.send_message(message)

        except Exception as error:

            if (
                is_temporary_gemini_error(error)
                and attempt < max_attempts
            ):

                error_text = str(error).lower()
                
                if "429" in error_text or "resource_exhausted" in error_text:
                    wait_time = 60
                else:
                    wait_time = attempt * 3

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds "
                    f"(attempt {attempt}/{max_attempts})..."
                )

                time.sleep(wait_time)

                continue

            raise


def generate_with_retry(
    prompt,
    max_attempts=3,
):
    """
    Generate standalone Gemini content.

    Used for counselor feedback and patient summaries.
    """

    for attempt in range(1, max_attempts + 1):

        try:

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

        except Exception as error:

            if (
                is_temporary_gemini_error(error)
                and attempt < max_attempts
            ):

                wait_time = attempt * 3

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds "
                    f"(attempt {attempt}/{max_attempts})..."
                )

                time.sleep(wait_time)

                continue

            raise


# ============================================================
# Preset Simulated Patient Profiles
# ============================================================

PATIENTS = {

    "Alex - College Anxiety": {

        "name": "Alex",

        "age": 21,

        "ethnicity": "Not specified",

        "background": (
            "College student"
        ),

        "concern": (
            "Academic anxiety and stress"
        ),

        "communication_style": (
            "Nervous and somewhat hesitant. Gives moderate-length "
            "answers but sometimes has difficulty explaining feelings."
        ),

        "symptoms": (
            "Trouble sleeping, excessive worrying about grades, "
            "difficulty concentrating, and fear of disappointing family."
        ),
    },

    "Jordan - Depression": {

        "name": "Jordan",

        "age": 40,

        "ethnicity": "Not specified",

        "background": (
            "Remote office worker"
        ),

        "concern": (
            "Depression and loss of motivation"
        ),

        "communication_style": (
            "Quiet and reserved. Usually gives short answers and does "
            "not volunteer much information unless asked."
        ),

        "symptoms": (
            "Low motivation, social withdrawal, fatigue, difficulty "
            "enjoying activities, and feeling disconnected from "
            "other people."
        ),
    },

    "Taylor - Work Burnout": {

        "name": "Taylor",

        "age": 32,

        "ethnicity": "Not specified",

        "background": (
            "High-achieving professional"
        ),

        "concern": (
            "Work-related burnout"
        ),

        "communication_style": (
            "Professional and controlled. Often minimizes emotional "
            "problems and tries to explain feelings logically."
        ),

        "symptoms": (
            "Exhaustion, irritability, difficulty separating work "
            "from personal life, poor sleep, and increasing frustration."
        ),
    },

    "Maya - Cultural Adjustment": {

        "name": "Maya",

        "age": 19,

        "ethnicity": "Not specified",

        "background": (
            "First-year college student living away from home"
        ),

        "concern": (
            "Cultural adjustment and loneliness"
        ),

        "communication_style": (
            "Friendly but cautious. Wants help but worries that other "
            "people will not understand her experiences."
        ),

        "symptoms": (
            "Homesickness, loneliness, difficulty making friends, "
            "anxiety in social situations, and uncertainty about "
            "fitting in."
        ),
    },
}


# ============================================================
# Custom Patient Functions
# ============================================================

def load_custom_patients():
    """
    Load custom simulated patients from custom_patients.json.
    """

    if not os.path.exists(CUSTOM_PATIENTS_FILE):
        return {}

    try:

        with open(
            CUSTOM_PATIENTS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):

        return {}


def save_custom_patients(custom_patients):
    """
    Save custom simulated patient profiles.
    """

    with open(
        CUSTOM_PATIENTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            custom_patients,
            file,
            indent=4,
            ensure_ascii=False,
        )


def get_all_patients():
    """
    Combine preset patients and saved custom patients.
    """

    all_patients = PATIENTS.copy()

    custom_patients = load_custom_patients()

    all_patients.update(custom_patients)

    return all_patients


# ============================================================
# Patient History Functions
# ============================================================

def load_patient_history():
    """
    Load previous counseling-session summaries.
    """

    if not os.path.exists(HISTORY_FILE):
        return {}

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except (json.JSONDecodeError, OSError):

        return {}


def save_patient_history(history):
    """
    Save patient counseling-session history.
    """

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )


def get_patient_history(patient_name):
    """
    Return all previous sessions for one patient.
    """

    history = load_patient_history()

    return history.get(
        patient_name,
        [],
    )


def add_patient_session(
    patient_name,
    summary,
):
    """
    Add a completed session to patient history.
    """

    history = load_patient_history()

    if patient_name not in history:
        history[patient_name] = []

    session_number = (
        len(history[patient_name]) + 1
    )

    history[patient_name].append(
        {
            "session": session_number,
            "summary": summary,
        }
    )

    save_patient_history(history)


def format_previous_history(
    previous_sessions,
):
    """
    Format previous sessions for Gemini.
    """

    if not previous_sessions:

        return (
            "No previous counseling sessions are available."
        )

    formatted_sessions = []

    for session in previous_sessions:

        formatted_sessions.append(
            f"Session {session['session']}:\n"
            f"{session['summary']}"
        )

    return "\n\n".join(
        formatted_sessions
    )


# ============================================================
# Patient Prompt
# ============================================================

def build_patient_prompt(
    patient,
    previous_sessions,
):
    """
    Create Gemini's simulated-patient instructions.
    """

    previous_history = format_previous_history(
        previous_sessions
    )

    return f"""
You are participating in a counselor training simulation.

You must role-play ONLY as the simulated patient described below.

PATIENT PROFILE

Name: {patient['name']}
Age: {patient['age']}
Ethnicity: {patient.get('ethnicity', 'Not specified')}
Background: {patient['background']}
Main concern: {patient['concern']}
Communication style: {patient['communication_style']}
Symptoms and struggles: {patient['symptoms']}

PREVIOUS COUNSELING SESSION HISTORY

{previous_history}

RULES

- Stay in character as this patient throughout the counseling session.
- Never act as the counselor.
- Never evaluate the counselor during the session.
- Do not provide therapy or counseling advice.
- Respond realistically based on the patient's background and emotions.
- Do not reveal all patient information immediately.
- Allow the counselor to learn more by asking appropriate questions.
- Keep most responses conversational and reasonably short.
- React naturally to empathy, questions, reassurance, and other responses.
- Use previous session history naturally when relevant.
- Do not repeat the entire previous session summary.
- If there is no previous history, treat this as the first session.
- Do not claim to be a real person.
- This is an educational simulation and not a real clinical encounter.
"""


# ============================================================
# Counselor Feedback Prompt
# ============================================================

def build_feedback_prompt(
    patient,
    transcript,
):
    """
    Create the Senior Clinical Supervisor evaluation prompt.
    """

    transcript_text = "\n".join(
        f"{entry['role'].upper()}: "
        f"{entry['message']}"
        for entry in transcript
    )

    return f"""
You are acting as a Senior Clinical Supervisor reviewing a counselor
training simulation.

This is an educational training exercise.

SIMULATED PATIENT

Name: {patient['name']}
Age: {patient['age']}
Ethnicity: {patient.get('ethnicity', 'Not specified')}
Background: {patient['background']}
Primary concern: {patient['concern']}

COUNSELING SESSION TRANSCRIPT

{transcript_text}

Evaluate ONLY the counselor's performance.

Provide feedback using exactly these sections:

## Counseling Session Feedback

### 1. Active Listening & Empathy
Score: X/10

Explain what the counselor did well and what could be improved.

### 2. Open-Ended Questioning
Score: X/10

Explain whether the counselor encouraged the patient to elaborate
instead of relying too heavily on yes/no questions.

### 3. Cultural Sensitivity
Score: X/10

Explain whether the counselor responded respectfully to the patient's
background, identity, values, and experiences.

### 4. Therapeutic Progress
Score: X/10

Explain whether the conversation explored the patient's concerns
and moved the session in a useful direction.

### 5. Professional Boundaries
Score: X/10

Explain whether the counselor maintained an appropriate professional
role and avoided judgment, inappropriate advice, or overstepping.

### 6. Safety & Ethics
Score: X/10

Explain whether the counselor appropriately recognized any safety
or ethical concerns that actually appeared in the conversation.

Do not invent a safety problem if none appeared.

### 7. Conversational Flow
Score: X/10

Explain whether the counselor's responses felt natural, relevant,
and connected to what the patient said.

### Overall Score
Score: X/10

### What the Counselor Did Well

Give 2-4 specific strengths based on the transcript.

### Areas for Improvement

Give 2-4 specific improvements based on the transcript.

### Example of an Improved Counselor Response

Choose one counselor response from the transcript that could have
been stronger.

Briefly explain why and provide a better example response.

Be constructive and specific.

Base the evaluation only on the transcript.

Do not diagnose the counselor or simulated patient.
"""


# ============================================================
# Patient Summary Prompt
# ============================================================

def build_summary_prompt(
    patient,
    transcript,
):
    """
    Generate information to remember for the patient's next session.
    """

    transcript_text = "\n".join(
        f"{entry['role'].upper()}: "
        f"{entry['message']}"
        for entry in transcript
    )

    return f"""
Summarize the following simulated counseling session for use during
the patient's next counseling session.

PATIENT

Name: {patient['name']}
Age: {patient['age']}
Primary concern: {patient['concern']}

COUNSELING SESSION TRANSCRIPT

{transcript_text}

Create a concise factual patient summary.

Include only information useful for continuity in a future session,
such as:

- Main concerns discussed
- Important personal or background information revealed
- Stressors or difficulties mentioned
- Feelings or symptoms discussed
- Coping strategies mentioned
- Goals or concerns for future sessions
- Important changes or progress discussed

Do NOT evaluate the counselor.

Do NOT give the counselor a score.

Do NOT add information that was not present in the conversation.

Keep the summary to approximately 1-2 short paragraphs.
"""


# ============================================================
# Create Gemini Patient Chat
# ============================================================

def create_patient_chat(
    patient,
    previous_sessions,
):
    """
    Create the Gemini chat used for one counseling session.
    """

    system_prompt = build_patient_prompt(
        patient,
        previous_sessions,
    )

    return client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.8,
        ),
    )


# ============================================================
# Start Chainlit Counseling Session
# ============================================================

@cl.on_chat_start
async def start():

    # Load preset and custom patients.
    all_patients = get_all_patients()

    patient_options = list(
        all_patients.keys()
    )

    if not patient_options:

        await cl.Message(
            content=(
                "No simulated patients are currently available."
            )
        ).send()

        return

    # --------------------------------------------------------
    # Patient Selection
    # --------------------------------------------------------

    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="Patient",
                label="Choose a Simulated Patient",
                values=patient_options,
                initial_index=0,
            )
        ]
    ).send()

    selected_patient = settings["Patient"]

    patient = all_patients[
        selected_patient
    ]

    previous_sessions = get_patient_history(
        patient["name"]
    )

    # --------------------------------------------------------
    # Store Chainlit Session Information
    # --------------------------------------------------------

    cl.user_session.set(
        "selected_patient",
        selected_patient,
    )

    cl.user_session.set(
        "patient",
        patient,
    )

    cl.user_session.set(
        "transcript",
        [],
    )

    cl.user_session.set(
        "session_ended",
        False,
    )

    cl.user_session.set(
        "previous_sessions",
        previous_sessions,
    )

    # --------------------------------------------------------
    # Create Gemini Patient
    # --------------------------------------------------------

    chat = create_patient_chat(
        patient,
        previous_sessions,
    )

    cl.user_session.set(
        "chat",
        chat,
    )

    # --------------------------------------------------------
    # Display Patient Profile
    # --------------------------------------------------------

    profile_message = (
        "## Simulated Counseling Session\n\n"
        f"**Patient:** {patient['name']}\n\n"
        f"**Age:** {patient['age']}\n\n"
        f"**Ethnicity:** "
        f"{patient.get('ethnicity', 'Not specified')}\n\n"
        f"**Background:** {patient['background']}\n\n"
        f"**Primary Concern:** {patient['concern']}\n\n"
    )

    if previous_sessions:

        profile_message += (
            f"**Previous Sessions:** "
            f"{len(previous_sessions)}\n\n"
            "### Previous Session Summary\n\n"
            f"{previous_sessions[-1]['summary']}\n\n"
        )

    else:

        profile_message += (
            "**Previous Sessions:** "
            "None - this is a new patient.\n\n"
        )

    profile_message += (
        "The simulated patient will begin the session.\n\n"
        "**When you are finished counseling, "
        "type `end session` to receive feedback.**"
    )

    await cl.Message(
        content=profile_message
    ).send()

    # --------------------------------------------------------
    # Patient Speaks First
    # --------------------------------------------------------

    if previous_sessions:

        opening_request = (
            "Begin this follow-up counseling session as the patient. "
            "Speak naturally about how things have been since the "
            "previous session or continue one relevant concern from "
            "the previous session. Do not summarize the entire "
            "previous session."
        )

    else:

        opening_request = (
            "Begin the counseling session as the patient. "
            "Say something natural that introduces why you came "
            "to counseling today. Do not explain that you are an "
            "AI or describe the role-play."
        )

    try:

        response = send_with_retry(
            chat,
            opening_request,
        )

        patient_message = response.text

        transcript = cl.user_session.get(
            "transcript"
        )

        transcript.append(
            {
                "role": "patient",
                "message": patient_message,
            }
        )

        cl.user_session.set(
            "transcript",
            transcript,
        )

        await cl.Message(
            author=patient["name"],
            content=patient_message,
        ).send()

    except Exception as error:

        await cl.Message(
            content=(
                "The simulated patient could not be started. "
                "Gemini may be temporarily unavailable.\n\n"
                f"Error: {str(error)}"
            )
        ).send()


# ============================================================
# Handle Counselor Messages
# ============================================================

@cl.on_message
async def handle_message(
    message: cl.Message,
):

    patient = cl.user_session.get(
        "patient"
    )

    chat = cl.user_session.get(
        "chat"
    )

    transcript = cl.user_session.get(
        "transcript"
    )

    session_ended = cl.user_session.get(
        "session_ended"
    )

    # --------------------------------------------------------
    # Verify Active Session
    # --------------------------------------------------------

    if not patient or not chat:

        await cl.Message(
            content=(
                "No simulated patient session "
                "is currently active."
            )
        ).send()

        return

    if session_ended:

        await cl.Message(
            content=(
                "This counseling session has ended. "
                "Start a new chat to begin another simulation."
            )
        ).send()

        return

    counselor_message = (
        message.content.strip()
    )

    if not counselor_message:
        return


    # ========================================================
    # End Counseling Session
    # ========================================================

    if counselor_message.lower() in {
        "end session",
        "end",
        "finish session",
        "finish",
    }:

        counselor_messages = [
            entry
            for entry in transcript
            if entry["role"] == "counselor"
        ]

        if not counselor_messages:

            await cl.Message(
                content=(
                    "There is not enough counselor conversation "
                    "to evaluate yet. Continue the counseling "
                    "session before ending it."
                )
            ).send()

            return

        cl.user_session.set(
            "session_ended",
            True,
        )

        await cl.Message(
            content=(
                "## Session Ended\n\n"
                "The counseling session is complete.\n\n"
                "Generating counselor feedback and saving "
                "the patient session summary..."
            )
        ).send()

        try:

            # ------------------------------------------------
            # Generate Counselor Feedback
            # ------------------------------------------------

            feedback_prompt = build_feedback_prompt(
                patient,
                transcript,
            )

            feedback_response = generate_with_retry(
                feedback_prompt
            )

            # ------------------------------------------------
            # Generate Patient Continuity Summary
            # ------------------------------------------------

            summary_prompt = build_summary_prompt(
                patient,
                transcript,
            )

            summary_response = generate_with_retry(
                summary_prompt
            )

            patient_summary = (
                summary_response.text
            )

            # ------------------------------------------------
            # Save Patient History
            # ------------------------------------------------

            add_patient_session(
                patient["name"],
                patient_summary,
            )

            # ------------------------------------------------
            # Display Counselor Feedback
            # ------------------------------------------------

            await cl.Message(
                author="Clinical Supervisor",
                content=feedback_response.text,
            ).send()

            await cl.Message(
                content=(
                    "### Patient History Saved\n\n"
                    f"A summary of this session with "
                    f"**{patient['name']}** has been saved.\n\n"
                    "If this patient is selected again, "
                    "the next session will use information "
                    "from the previous counseling session."
                )
            ).send()

        except Exception as error:

            # Allow another attempt if Gemini failed.
            cl.user_session.set(
                "session_ended",
                False,
            )

            await cl.Message(
                content=(
                    "The session could not be finalized because "
                    "Gemini is currently unavailable or another "
                    "API error occurred.\n\n"
                    "Your current session has not been discarded. "
                    "You can try `end session` again.\n\n"
                    f"Error: {str(error)}"
                )
            ).send()

        return


    # ========================================================
    # Normal Counseling Conversation
    # ========================================================

    transcript.append(
        {
            "role": "counselor",
            "message": counselor_message,
        }
    )

    cl.user_session.set(
        "transcript",
        transcript,
    )

    try:

        response = send_with_retry(
            chat,
            counselor_message,
        )

        patient_message = response.text

        transcript.append(
            {
                "role": "patient",
                "message": patient_message,
            }
        )

        cl.user_session.set(
            "transcript",
            transcript,
        )

        await cl.Message(
            author=patient["name"],
            content=patient_message,
        ).send()

    except Exception as error:

        await cl.Message(
            content=(
                "The simulated patient could not respond. "
                "Gemini may be temporarily experiencing "
                "high demand.\n\n"
                "Your counseling session is still active, "
                "so you can try sending your message again.\n\n"
                f"Error: {str(error)}"
            )
        ).send()