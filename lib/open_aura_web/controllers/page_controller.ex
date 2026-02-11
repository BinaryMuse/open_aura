defmodule OpenAuraWeb.PageController do
  use OpenAuraWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
