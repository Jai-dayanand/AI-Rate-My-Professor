import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAI } from "openai"; // Ensure correct import

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // Use your environment variable here
});

const systemPrompt = `
You are an AI assistant designed to help students find the best professors according to the specific queries. Your task is to provide students with the top 3 professors on the basis of inputs.
Use the Retrieval-Augmented Generation (RAG) model to retrieve and rank professors who match the query provided by the user.

Receive Query: Listen carefully to the user's query to understand their requirements (e.g., subject, teaching style).

Retrieve Information: Use the RAG model to search for professors who best fit the criteria described in the query.

Rank Professors: Determine the top 3 professors based on relevance, ratings, and user reviews.

Respond with Details: Provide the user with a list of the top 3 professors, including their names, subjects they teach, and any other relevant information such as ratings or notable reviews.
`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString = "Returned results from vector db (done automatically):";
  results.matches.forEach((match) => {
    resultString += `
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    `;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  const completion = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: 'user', content: lastMessageContent }
    ],
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    }
  });

  return new NextResponse(stream);
}
