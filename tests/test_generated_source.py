import parser.generated_source as generated_source


def test_signature():
    input_source = f"""
This is a random thing.

Put your slug here: {generated_source.signature_token()}

Here are some more things:

    - Machine Generated
    - More Machine Gnerated
{generated_source.begin_manual_section('foo')}
    - You can edit here!
{generated_source.end_manual_section()}
    - But not here.
{generated_source.begin_manual_section('bar')}
    - You can edit here too!
{generated_source.end_manual_section()}
    - Also not here.
"""
    signed = generated_source.sign_generated_source(input_source)
    assert signed != input_source
    assert generated_source.is_signed(signed)
    assert generated_source.validate_signature(signed)


def test_manual_changes():
    input_source = f"""
This is a random thing.

Put your slug here: {generated_source.signature_token()}

Here are some more things:

    - Machine Generated
    - More Machine Gnerated
{generated_source.begin_manual_section('foo')}
    - XXXXX
{generated_source.end_manual_section()}
    - But not here.
"""
    signed = generated_source.sign_generated_source(input_source)
    modified = signed.replace("XXXXX", "YYYYY")
    assert modified != signed

    assert generated_source.is_signed(modified)
    assert generated_source.validate_signature(modified)


def test_bad_changes():
    input_source = f"""
This is a random thing.

Put your slug here: {generated_source.signature_token()}

Here are some more things:

    - Machine Generated
    - More Machine Gnerated
{generated_source.begin_manual_section('foo')}
    - XXXXX
{generated_source.end_manual_section()}
    - ZZZZZ
"""
    signed = generated_source.sign_generated_source(input_source)
    modified = signed.replace("ZZZZZ", "YYYYY")
    assert modified != signed

    assert generated_source.is_signed(modified)
    assert not generated_source.validate_signature(modified)


def test_merge_changes():
    original_source = f"""
A
// {generated_source.begin_manual_section('foo')}
B
// {generated_source.end_manual_section()}
C
// {generated_source.begin_manual_section('bar')}
D
// {generated_source.end_manual_section()}
"""
    new_source = f"""
E
// {generated_source.begin_manual_section('bar')}
F
// {generated_source.end_manual_section()}
// {generated_source.begin_manual_section('foo')}
G
// {generated_source.end_manual_section()}
H
"""

    merged = generated_source.merge_existing(original_source, new_source)
    assert (
        merged
        == f"""
E
// {generated_source.begin_manual_section('bar')}
D
// {generated_source.end_manual_section()}
// {generated_source.begin_manual_section('foo')}
B
// {generated_source.end_manual_section()}
H
"""
    )
