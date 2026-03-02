from unittest import mock

import pytest


@mock.patch("loader.PyPDFLoader")
@mock.patch("loader.glob.glob")
def test_load_and_split_returns_chunks(mock_glob, mock_pdf_loader):
    mock_glob.return_value = ["/fake/path/doc1.pdf", "/fake/path/doc2.pdf"]

    fake_doc = mock.MagicMock()
    fake_doc.page_content = "Some text content that is long enough to be meaningful."
    fake_doc.metadata = {}
    mock_pdf_loader.return_value.load.return_value = [fake_doc]

    from loader import load_and_split
    chunks = load_and_split()

    assert len(chunks) > 0
    assert mock_pdf_loader.call_count == 2


@mock.patch("loader.glob.glob", return_value=[])
def test_load_and_split_raises_when_no_pdfs(mock_glob):
    from loader import load_and_split

    with pytest.raises(FileNotFoundError, match="No PDFs found"):
        load_and_split()


@mock.patch("loader.PyPDFLoader")
@mock.patch("loader.glob.glob")
def test_chunk_count(mock_glob, mock_pdf_loader):
    mock_glob.return_value = ["/fake/doc.pdf"]

    # Create a document with enough content to produce multiple chunks
    fake_doc = mock.MagicMock()
    fake_doc.page_content = "word " * 500  # ~2500 chars, should split into multiple chunks
    fake_doc.metadata = {}
    mock_pdf_loader.return_value.load.return_value = [fake_doc]

    from loader import load_and_split
    chunks = load_and_split()

    # With default chunk_size=1000, 2500 chars should yield more than 1 chunk
    assert len(chunks) >= 2
