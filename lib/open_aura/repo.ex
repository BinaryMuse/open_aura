defmodule OpenAura.Repo do
  use Ecto.Repo,
    otp_app: :open_aura,
    adapter: Ecto.Adapters.Postgres
end
