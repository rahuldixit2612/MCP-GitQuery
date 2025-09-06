import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def run():
    print("🚀 Starting GitHub Repo NLP Workflow...")

    async with sse_client(url="http://localhost:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("🛠️ Available tools:", [name for name, _ in tools])

            print("📥 Step 1: Extracting...")
            step1 = await session.call_tool("extract_documents", {
                "owner": "rahuldixit2612",
                "repo": "mcp_poc",
                "branch": "main"
            })
            print("✅", step1.content[0].text)

            print("\n🔧 Step 2: Preprocessing...")
            step2 = await session.call_tool("preprocess_text")
            print("✅", step2.content[0].text)

            print("\n💡 Step 3: Generating Embeddings...")
            step3 = await session.call_tool("generate_embeddings")
            print("✅", step3.content[0].text)

            print("\n📦 Step 4: Creating FAISS Index...")
            step4 = await session.call_tool("create_faiss_index")
            print("✅", step4.content[0].text)


            import json

            print("\n🔍 Step 5: Multi-query FAISS Search...")
            queries = [
                "What method is used to close the current browser window?",
                "What method is used to Quit the WebDriver instance and cleanup?"
            ]
            step5 = await session.call_tool("multi_query_search", {"queries": queries})
            results = json.loads(step5.content[0].text)["results"]

            for result in results:
                print(f"\n🔎 Query: {result['query']}")
                for match in result["matches"]:
                    print(f"📄 Match (distance={match['distance']:.4f}): {match['text'][:150]}...")

if __name__ == "__main__":
    asyncio.run(run())
