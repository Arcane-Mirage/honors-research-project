"""tests/test_system_prompt.py

The extra system-prompt text: placeholders, which prompts get it, how it is recorded, its switches.
"""

from helpers import ScriptedProvider

from csr_lab.config import SystemPromptSettings, apply_experiment, load_config, load_experiment
from csr_lab.llm.interface import CallRecorder, StructuredLLM
from csr_lab.pipelines.run import run_analysis
from csr_lab.prompts import system_prompt
from csr_lab.prompts.references import ReferenceExtractionResponse

NAVISTAR = SystemPromptSettings(
    enabled=True,
    supplier="Bendix",
    customer="Navistar",
    customer_aliases=["International"],
    ambiguous_aliases=["International"],
)


def test_placeholders_and_options():
    text = system_prompt.render(NAVISTAR)
    assert text.splitlines()[0] == "- You read these documents for Bendix, a direct supplier. Its customers include Navistar."
    assert '- Navistar is also called "International".' in text
    assert '"International" means Navistar only when it names the company' in text
    assert "{" not in text and "brand" not in text.lower()
    assert "alike" not in text  # one customer: nothing to treat alike
    # The customer list is not complete: nothing is skipped unless a company is listed as a non-customer.
    assert "Other companies may also be customers of Bendix" in text and "Ignore text" not in text
    listed = NAVISTAR.model_copy(update={"non_customer_names": {"Brennic": []}})
    assert "Ignore text that applies only to Brennic; they are not customers of Bendix." in system_prompt.render(listed)
    keep = system_prompt.render(listed.model_copy(update={"non_customers": "keep"}))
    assert "Analyse text for Brennic too" in keep and "Ignore text" not in keep
    plain = system_prompt.render(SystemPromptSettings(enabled=True, supplier="Bendix", customer="Navistar"))
    assert "also called" not in plain and "ordinary word" not in plain
    assert system_prompt.render(NAVISTAR.model_copy(update={"enabled": False})) is None
    assert system_prompt.fingerprint(NAVISTAR.model_copy(update={"enabled": False})) == "off"
    assert system_prompt.fingerprint(listed) != system_prompt.fingerprint(listed.model_copy(update={"non_customers": "keep"}))


def test_every_customer_is_named_and_treated_alike():
    text = system_prompt.render(load_config().system_prompt)
    first = text.splitlines()[0]
    for name in ("Navistar", "MAN", "Scania", "VWTB", "Volkswagen", "Audi", "Porsche", "ŠKODA AUTO", "Bentley", "SEAT"):
        assert name in first
    assert "TRATON" not in first  # an issuer, not a customer
    assert "Treat every one of these customers alike" in text
    assert '"International" means Navistar only when it names the company' in text
    assert "other customers" not in text and "Ignore text" not in text  # no company is assumed not to be a customer


def test_text_goes_only_to_the_listed_tasks(config):
    provider = ScriptedProvider(lambda task, payload, request: {"results": []})
    llm = StructuredLLM(provider, config.llm, CallRecorder(), extra_system="EXTRA-TEXT", extra_system_tasks=system_prompt.TASKS)
    for task in ("reference_extraction", "semantic_chunking"):
        llm.call(task=task, stage=task, system="SYSTEM", user="<input_json>{}</input_json>", response_model=ReferenceExtractionResponse, prompt_version="t")
    systems = [r.messages[0]["content"] for r in provider.requests]
    assert "EXTRA-TEXT" in systems[0] and "EXTRA-TEXT" not in systems[1]


def test_recorded_in_results_and_off_in_experiments(simple_docs, config):
    on = config.model_copy(update={"system_prompt": NAVISTAR})
    off = config.model_copy(update={"system_prompt": NAVISTAR.model_copy(update={"enabled": False})})
    a = run_analysis(simple_docs, strategy="staged", config=on)
    b = run_analysis(simple_docs, strategy="staged", config=off)
    assert a.metadata.prompt_versions["system_prompt"].startswith(system_prompt.VERSION)
    assert b.metadata.prompt_versions["system_prompt"] == "off"
    assert a.analysis_id != b.analysis_id
    # default.yaml reads for Bendix; experiments read for the dataset's reader (its manifest)
    assert "Navistar" in load_config().system_prompt.customer_names
    experiment = apply_experiment(load_config(), load_experiment("demo_mock"))
    assert experiment.system_prompt.enabled and experiment.system_prompt.customer_names == ["Global Motors"]
    assert experiment.system_prompt.non_customer_names == {"Brennic": ["Brennic Trucks"], "Ostra": ["Ostra AB"]}
    assert experiment.evaluation.customer == "Global Motors"
