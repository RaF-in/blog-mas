"""Simulated knowledge base with hardcoded topics and lookup function."""

_KNOWLEDGE_BASE: dict[str, str] = {
    "Mediterranean diet": (
        "The Mediterranean diet is a dietary pattern inspired by the traditional eating habits "
        "of countries bordering the Mediterranean Sea, particularly Greece, Italy, and Spain. "
        "It emphasizes the consumption of fruits, vegetables, whole grains, legumes, nuts, and "
        "olive oil as the primary source of fat. This dietary approach has been extensively "
        "studied and is associated with numerous health benefits, including reduced risk of "
        "cardiovascular disease, improved cognitive function, and better weight management."
        "\n\n"
        "Central to the Mediterranean diet is the moderate consumption of fish and poultry, "
        "while red meat and processed foods are kept to a minimum. Dairy products, particularly "
        "yogurt and cheese, are consumed in low to moderate amounts. The diet also allows for "
        "the moderate intake of red wine during meals, which some studies suggest may contribute "
        "to heart health due to its antioxidant properties."
        "\n\n"
        "Research published in the New England Journal of Medicine demonstrated that the "
        "Mediterranean diet supplemented with extra-virgin olive oil or nuts reduced the "
        "incidence of major cardiovascular events by approximately 30 percent compared to a "
        "low-fat control diet. The diet's anti-inflammatory properties, largely attributed to "
        "its high content of omega-3 fatty acids and polyphenols, are believed to be key "
        "mechanisms behind its protective effects against chronic disease."
        "\n\n"
        "Beyond its nutritional profile, the Mediterranean diet embodies a holistic approach "
        "to eating that includes social dining, seasonal food selection, and home cooking. "
        "It has been recognized by UNESCO as an Intangible Cultural Heritage of Humanity, "
        "reflecting its significance not only as a dietary pattern but as a cultural practice "
        "that promotes community, sustainability, and a balanced relationship with food."
    ),
    "Artificial intelligence": (
        "Artificial intelligence (AI) refers to the simulation of human intelligence processes "
        "by computer systems, encompassing learning, reasoning, problem-solving, perception, and "
        "language understanding. The field was formally established at the Dartmouth Conference "
        "in 1956 and has since evolved through several cycles of optimism and setbacks, often "
        "referred to as AI summers and winters."
        "\n\n"
        "Modern AI is largely driven by machine learning, a subset of techniques that enable "
        "systems to improve their performance on tasks through exposure to data rather than "
        "through explicit programming. Deep learning, which uses multi-layered neural networks, "
        "has produced breakthroughs in image recognition, natural language processing, and game "
        "playing. Large language models such as GPT and BERT have demonstrated remarkable "
        "capabilities in generating human-like text, translating languages, and answering "
        "complex questions."
        "\n\n"
        "The ethical implications of AI deployment are a growing area of concern and research. "
        "Issues include algorithmic bias, in which models perpetuate or amplify societal "
        "prejudices present in training data; job displacement due to automation; the "
        "environmental cost of training large models; and questions about accountability when "
        "AI systems make consequential decisions in areas such as healthcare, criminal justice, "
        "and autonomous vehicles."
        "\n\n"
        "Looking ahead, researchers are pursuing artificial general intelligence (AGI), systems "
        "that could match or exceed human cognitive abilities across a wide range of domains. "
        "While current AI systems excel at narrow, well-defined tasks, the path to AGI remains "
        "uncertain and is the subject of active debate regarding timelines, safety, and the "
        "societal transformations such technology would bring."
    ),
    "Climate change": (
        "Climate change refers to long-term shifts in global temperatures and weather patterns, "
        "driven predominantly by human activities since the Industrial Revolution. The burning "
        "of fossil fuels such as coal, oil, and natural gas releases greenhouse gases, primarily "
        "carbon dioxide and methane, into the atmosphere. These gases trap heat from the sun, "
        "causing the Earth's average surface temperature to rise, a phenomenon known as the "
        "greenhouse effect."
        "\n\n"
        "The consequences of climate change are far-reaching and increasingly visible. Rising "
        "sea levels threaten coastal communities and island nations, while extreme weather events "
        "such as hurricanes, droughts, and wildfires have become more frequent and intense. Arctic "
        "ice is melting at unprecedented rates, disrupting ocean currents and ecosystems. "
        "Biodiversity loss is accelerating as species struggle to adapt to rapidly changing "
        "habitats, with some projections suggesting that up to one million species face extinction."
        "\n\n"
        "International efforts to address climate change include the Paris Agreement of 2015, "
        "in which nearly 200 nations committed to limiting global warming to well below 2 degrees "
        "Celsius above pre-industrial levels, with an aspirational target of 1.5 degrees. "
        "Achieving these goals requires a rapid transition from fossil fuels to renewable energy "
        "sources such as solar, wind, and hydroelectric power, along with improvements in energy "
        "efficiency, carbon capture technology, and sustainable land use practices."
        "\n\n"
        "Despite growing awareness and technological advances, global greenhouse gas emissions "
        "continue to rise. The gap between current commitments and the reductions needed to meet "
        "Paris Agreement targets remains substantial. Climate scientists emphasize that every "
        "fraction of a degree of warming matters, and that decisive action in the current decade "
        "is critical to avoiding the most catastrophic outcomes for future generations."
    ),
    "Space exploration": (
        "Space exploration is the investigation and exploration of outer space using astronomy "
        "and space technology. Beginning with the Soviet Union's launch of Sputnik 1 in 1957, "
        "the Space Age has seen remarkable achievements including human Moon landings, robotic "
        "missions to every planet in the solar system, and the construction of the International "
        "Space Station as a permanently crewed orbital laboratory."
        "\n\n"
        "The current era of space exploration is characterized by increasing commercial "
        "participation. Companies such as SpaceX, Blue Origin, and Rocket Lab have dramatically "
        "reduced the cost of reaching orbit through reusable rocket technology. SpaceX's Falcon 9 "
        "and Starship vehicles represent a paradigm shift in launch economics, enabling more "
        "frequent and affordable access to space for scientific missions, satellite deployment, "
        "and eventually crewed missions to Mars."
        "\n\n"
        "Scientific milestones continue to drive public interest and scientific advancement. "
        "NASA's Perseverance rover is exploring Mars for signs of ancient microbial life and "
        "collecting samples for future return to Earth. The James Webb Space Telescope, launched "
        "in 2021, is providing unprecedented views of the universe in infrared light, revealing "
        "details about the formation of the earliest galaxies, the atmospheres of exoplanets, "
        "and the lifecycle of stars."
        "\n\n"
        "Looking forward, the coming decades promise even more ambitious endeavors. NASA's "
        "Artemis program aims to establish a sustained human presence on the Moon as a stepping "
        "stone to Mars. Plans for lunar bases, asteroid mining, and space tourism are moving from "
        "science fiction toward engineering reality. International cooperation, particularly "
        "through the European Space Agency and emerging space programs in India, China, and the "
        "United Arab Emirates, is expanding humanity's collective reach into the cosmos."
    ),
    "Mental health": (
        "Mental health encompasses emotional, psychological, and social well-being, affecting how "
        "people think, feel, and act. It plays a critical role in determining how individuals "
        "handle stress, relate to others, and make choices. Mental health disorders, including "
        "depression, anxiety, bipolar disorder, and schizophrenia, affect hundreds of millions of "
        "people worldwide and represent a leading cause of disability."
        "\n\n"
        "The World Health Organization estimates that approximately one in eight people globally "
        "lives with a mental health condition. Depression alone affects more than 280 million "
        "people and is a major contributor to the global burden of disease. Despite the prevalence "
        "of these conditions, a significant treatment gap persists, particularly in low- and "
        "middle-income countries where access to mental health professionals and evidence-based "
        "therapies remains severely limited."
        "\n\n"
        "Stigma and discrimination continue to be major barriers to mental health care. Many "
        "individuals avoid seeking help due to fear of social judgment, workplace repercussions, "
        "or cultural taboos surrounding mental illness. Public health campaigns and celebrity "
        "advocacy have helped normalize conversations about mental health in recent years, but "
        "systemic changes in healthcare policy, insurance coverage, and workplace accommodations "
        "are still needed to ensure equitable access to treatment."
        "\n\n"
        "Advances in neuroscience, digital therapeutics, and telehealth are expanding the toolkit "
        "for mental health treatment. Cognitive behavioral therapy, mindfulness-based interventions, "
        "and pharmacological treatments remain cornerstones of care, while emerging approaches such "
        "as psychedelic-assisted therapy, transcranial magnetic stimulation, and AI-driven mental "
        "health chatbots show promise for treatment-resistant conditions and improving access in "
        "underserved populations."
    ),
}


def get_available_topics() -> list[str]:
    return list(_KNOWLEDGE_BASE.keys())


def lookup_topic(query: str) -> str | None:
    if not query or not query.strip():
        return None
    query_lower = query.lower().strip()
    # Exact case-insensitive match first
    for topic, content in _KNOWLEDGE_BASE.items():
        if topic.lower() == query_lower:
            return content
    # Partial/fuzzy match: query is a substring of a topic name
    for topic, content in _KNOWLEDGE_BASE.items():
        if query_lower in topic.lower():
            return content
    return None
