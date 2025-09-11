# **TRUELENS**
### *AI Browser for finding human art*
*![GPTOSS](https://img.shields.io/badge/GPTOSS_Powered_Agent-000?style=for-the-badge&logo=openai&logoColor=white)*

## Technologies Used
![React](https://img.shields.io/badge/React-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TailwindCSS](https://img.shields.io/badge/Tailwindcss-00a6f4?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)


## API Calls
![HuggingFace](https://img.shields.io/badge/Huggingface-%2320232a?style=for-the-badge&logo=huggingface&logoColor=#FFD21E)
![Exa](https://img.shields.io/badge/Exa_AI-0137E1?style=for-the-badge)
![Clarifai](https://img.shields.io/badge/Clarifai-0270e0?style=for-the-badge&logo=clarifai&logoColor=white)

## Instructions

  #### Install the dependencies
  ` 
    git clone https://github.com/TrueLensAI/openai-hackathon &&
    cd openai-hackathon &&
    npm install &&
    cd backend &&
    pip install -r requirements.txt
  `
  #### Open a terminal window and run:
  ` npm run dev `

  #### In a seperate terminal window run `main.py` or click the run button with VS Code

#### Visit https://localhost:5173 and start typing!



## What is TrueLens?

TrueLens is an experimental search agent created by 4 College Students trying to answer the **question**?

### **Can AI serve as a tool to expand access to human-created visual content?**

#### Problem:
Throughout the internet, artists and designers were one of the first professionals to experience the fear that *'AI will take my job'*.
Now as college students, we all constantly hear the discussion around AI surround the belief that it's removing value from our creative output. Sadly, we can clearly agree with a lot of the concerns that arise in this debate.
Nevertheless, we still believe AI Agents present a new possibility one to **empower artists** and help them reach broader markets.

Our **hypothesis**: Yes!

Our **conclusion**: It's a little more complicated than we thought, but we are on the right track.

#### Solution:
TrueLens uses GPT-OSS to process the user's request helping them clearly define the design they are trying to find.
Once satisfied it calls a two part search tool:
1. Scraping marketplaces using EXA AI to identify designs currently for sale
2. Vectorizes both the image and the prompt using a finetuned version of OpenClip hosted in Clarifai and returns a similarity score



## Potential Changes/Problems we faced

While this MVP doesn't perform as we expected it did help us create a structural framework for the agent, which could be improved upon in the future.

*Alternatives for Scraping*
1. While Exa AI was incredibly easy to implement, using an finetuned multimodal model capable identifying the right website based on the "image" rather than the "keyword" would present a drastic improvement.
2. Another alternative we would hope to look into is a traditional web scraping API with the GPT-OSS model tailoring the request.





## Screenshots
_Landing Page_
<img width="1430" height="697" alt="Screen Shot 2025-09-11 at 1 55 03 AM" src="https://github.com/user-attachments/assets/34467b60-a545-43b0-bca9-12be2b84f0ea" />
<img width="1420" height="671" alt="Screen Shot 2025-09-11 at 1 55 18 AM" src="https://github.com/user-attachments/assets/13a0128f-b744-4d7a-ae7a-ff40e286f198" />


_Input Page_
<img width="1423" height="644" alt="Screen Shot 2025-09-11 at 1 58 12 AM" src="https://github.com/user-attachments/assets/b397ccd7-e352-4d61-a82f-16bc3f2a1284" />



<img width="1433" height="696" alt="Screen Shot 2025-09-11 at 1 56 06 AM" src="https://github.com/user-attachments/assets/820f743f-488b-44d5-9ce5-5e8abd6f589c" />


<img width="1212" height="693" alt="Screen Shot 2025-09-11 at 1 56 16 AM" src="https://github.com/user-attachments/assets/8f1b5f91-00d8-4410-a117-948128da6c07" />
