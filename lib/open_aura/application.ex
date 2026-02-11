defmodule OpenAura.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      OpenAuraWeb.Telemetry,
      OpenAura.Repo,
      {DNSCluster, query: Application.get_env(:open_aura, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: OpenAura.PubSub},
      # Start a worker by calling: OpenAura.Worker.start_link(arg)
      # {OpenAura.Worker, arg},
      # Start to serve requests, typically the last entry
      OpenAuraWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: OpenAura.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    OpenAuraWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
